/*
 * 検索のトレース(query Lambda の応答)→ 図のデータ(列・ノード・エッジ・チャンクのバッジ・再生の段階)。
 *
 * 画面(index.html)と node の両方から使う。
 * DOM には触らない純粋な変換だけを置く。外部ライブラリは使わない。
 *
 * 列: 0 列目 = 入口(trace.entry。選外も含む)、h 列目 = ホップ h で評価した隣候補(hops[h].neighbor_scores)。
 *     同じエンティティでも列が違えば別のノード(id は "列:エンティティ ID")。
 * エッジ: hops[h].neighbor_scores の各候補。from(前の列で選ばれたノード)→ entity(この列)。
 *     矢印はテンプレートの向き(source → target)。direction が out なら from → entity(右向き)、
 *     in なら entity → from(左向き)。sym(対称な関係)は向きなし(arrow="none"、
 *     線は from → entity に引くが矢印は付けず、ラベルに「⇄」)。also_from(別の frontier から同じ関係で
 *     来た経路)も別エッジにする。
 * バッジ: trace.chunks の {id, hop, via_entity} を「hop 列の via_entity ノード」に付ける。
 *     also_via(後のホップで別の経路から再び出会ったもの)は薄いバッジ(also: true)。
 *     引用されたチャンク(citations の chunk_ids)は cited、評価の質問の required_chunks は required。
 * 下限未満: neighbor_scores の excluded === "below_min"(スコアが min_neighbor_score 未満で選外)は
 *     エッジと(その列の候補が全部下限未満の)ノードに belowMin: true を付ける。見た目は普通の選外と同じ。
 * 再生: 列ごとに 4 段階(候補が現れる → 選ぶ → チャンク取得 → 十分性)+ 最後に stop_reason。
 * 経路: 選んで進んだエッジ(selected)= 通過経路。そのうち引用されたチャンクを**最初に取った**ノード
 *     (chunks[] の hop/via_entity。also_via は含めない)から選んだエッジを入口までさかのぼったものが
 *     「引用への経路」(citedPath: true)。
 * 1 ノードの上限: hops[h].node_chunk_counts(進んだノードの候補チャンク数)と capped_chunk_ids(上限
 *     max_chunks_per_node からあふれた分)を h 列のノードに appearanceCount / cappedChunkIds として付ける。
 * 1 ホップの上限: hops[h].hop_capped_by_entity(1 ホップで足せる max_chunks_per_hop 件からあふれた分。
 *     スコア順)を h 列のノードに hopCappedChunkIds、hops[h].hop_capped_chunk_ids を列に hopCappedChunkIds、
 *     params.max_chunks_per_hop を maxChunksPerHop として付ける(古いトレースは空・null)。
 * チャンクの取り方: hops[h].chunk_source("appearance" = 登場チャンク上位 / "edge_evidence" = たどった関係の
 *     根拠チャンク。古いトレースは無いので null)を、h 列で進んだノードに chunkSource として付ける。
 *     バッジには chunks[].evidence_of(どの関係の根拠として取ったか)を evidenceOf として付ける。
 * 入口の見つけ方: trace.entry[].via("dictionary" = 辞書の候補への「書かれているか」 /
 *     "semantic" = 入口 0 件のときの意味での聞き直し。古いトレースは無いので null)を 0 列目のノードに via として
 *     付ける。trace.entry_fallback_used / entry_fallback は entryFallback({used, foundBySemantic, selectedTypes,
 *     threshold, nQuestions})にまとめる(画面の見出しのタグとポップアップ用)。
 * 時間と費用の内訳: costBreakdown(trace, local) が Bedrock(回答生成)・Jev・DynamoDB の時間・費用・
 *     回数と、合計に占める割合を返す(画面の結果見出しのチップと横棒用)。
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.GraphModel = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var STEPS_PER_COLUMN = 4;
  var STEP_KIND = ["candidates", "select", "chunks", "sufficiency"];
  var PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1",
                 "#76b7b2", "#edc948", "#9c755f", "#ff9da7", "#bab0ac"];

  function nodeKey(col, entity) {
    return col + ":" + entity;
  }

  function typeColors(master) {
    var colors = {};
    var types = (master && master.node_types) || [];
    types.forEach(function (t, i) {
      colors[t.name] = PALETTE[i % PALETTE.length];
    });
    return colors;
  }

  function entityIndex(master) {
    var idx = {};
    ((master && master.entities) || []).forEach(function (e) {
      idx[e.id] = e;
    });
    return idx;
  }

  function citedSet(trace) {
    var s = {};
    (trace.citations || []).forEach(function (c) {
      (c.chunk_ids || []).forEach(function (id) {
        s[String(id).toLowerCase()] = true;
      });
    });
    return s;
  }

  function stepOf(col, kind) {
    return col * STEPS_PER_COLUMN + STEP_KIND.indexOf(kind);
  }

  /**
   * @param trace   query Lambda の応答(ok:true のもの。entry / hops / chunks / citations / stop_reason)
   * @param master  data/master.json(entities / node_types)
   * @param opts    {required: ["dc-3", ...], showRejected: true}
   * @returns {columns, nodes, edges, badges, steps, totalSteps, stopReason, haltedAt, typeColors}
   */
  function buildGraph(trace, master, opts) {
    opts = opts || {};
    var showRejected = opts.showRejected !== false;
    var required = {};
    (opts.required || []).forEach(function (id) {
      required[String(id).toLowerCase()] = true;
    });
    var ents = entityIndex(master);
    var cited = citedSet(trace);
    var hops = trace.hops || [];

    var columns = [];
    var nodes = [];
    var nodeById = {};
    var edges = [];
    var badges = [];

    function info(entity) {
      var e = ents[entity] || {};
      return {name: e.name || entity, type: e.type || "不明"};
    }

    function addNode(col, entity, fields) {
      var key = nodeKey(col, entity);
      var node = nodeById[key];
      if (node) {
        // 同じ列に複数の経路で現れた: 選ばれた・高いスコアのほうを残す
        if (fields.selected) node.selected = true;
        // 下限未満 = この列のどの候補も下限未満(1 件でも下限以上なら外す)
        if (!fields.belowMin) node.belowMin = false;
        if (fields.score != null && (node.score == null || fields.score > node.score)) node.score = fields.score;
        if (fields.via && !node.via) node.via = fields.via;
        return node;
      }
      var i = info(entity);
      node = {
        id: key, entity: entity, name: i.name, type: i.type, col: col, row: 0,
        score: fields.score == null ? null : fields.score,
        selected: !!fields.selected,
        belowMin: !!fields.belowMin,
        via: fields.via || null,
        appearStep: stepOf(col, "candidates"),
        selectStep: stepOf(col, "select"),
        appearanceCount: null,
        cappedChunkIds: [],
        hopCappedChunkIds: [],
        chunkSource: null,
        chunks: []
      };
      nodeById[key] = node;
      nodes.push(node);
      return node;
    }

    // ---- 0 列目: 入口
    var entry = trace.entry || [];
    var selected0 = {};
    ((hops[0] && hops[0].selected) || []).forEach(function (id) { selected0[id] = true; });
    entry.forEach(function (e) {
      var sel = !!e.selected || !!selected0[e.id];
      if (!sel && !showRejected) return;
      addNode(0, e.id, {score: e.score, selected: sel, via: e.via || null});
    });
    // entry に無いが hops[0].selected にある(古いトレースなど)ものも出す
    Object.keys(selected0).forEach(function (id) {
      addNode(0, id, {score: null, selected: true});
    });
    if (entry.length || hops.length) {
      columns.push(makeColumn(0, hops[0]));
    }

    // ---- 1 列目以降: 隣候補
    for (var h = 1; h < hops.length; h++) {
      var hop = hops[h];
      var chosen = {};
      (hop.selected || []).forEach(function (id) { chosen[id] = true; });
      (hop.neighbor_scores || []).forEach(function (c, i) {
        var sel = !!chosen[c.entity];
        // 選ばれたノードに入るエッジのうち「選んだ候補」は、そのノードで最高スコアの候補
        if (!sel && !showRejected) return;
        var below = c.excluded === "below_min";
        addNode(h, c.entity, {score: c.score, selected: sel, belowMin: below});
        var sources = [{from: c.from, direction: c.direction, also: false}];
        (c.also_from || []).forEach(function (a) {
          sources.push({from: a.from, direction: a.direction, also: true});
        });
        sources.forEach(function (src, j) {
          var fromKey = nodeKey(h - 1, src.from);
          if (!nodeById[fromKey]) {
            // frontier のノードが前の列に無い(showRejected=false でも frontier は必ず選ばれている)
            addNode(h - 1, src.from, {score: null, selected: true});
          }
          var out = src.direction !== "in";            // out と sym は from → entity の向きに引く
          var sym = src.direction === "sym";
          edges.push({
            id: "e" + h + "-" + i + "-" + j,
            hop: h,
            from: fromKey,
            to: nodeKey(h, c.entity),
            edge: c.edge,
            direction: src.direction,
            // テンプレートの source → target。out: from → entity / in: entity → from
            // sym: 向きなし(source = from・target = entity は線を引く都合。テンプレートにも (自分, 相手) で入れている)
            source: out ? fromKey : nodeKey(h, c.entity),
            target: out ? nodeKey(h, c.entity) : fromKey,
            arrow: sym ? "none" : out ? "right" : "left",
            score: c.score == null ? null : c.score,
            sentence: c.sentence || "",
            also: src.also,
            belowMin: below,
            selected: false,
            appearStep: stepOf(h, "candidates"),
            selectStep: stepOf(h, "select")
          });
        });
      });
      // 選んで進んだエッジ = 選ばれたノードごとにスコア最高の候補(also_from も含めて同じ候補なら全部)
      markSelectedEdges(edges, h, chosen);
      columns.push(makeColumn(h, hop));
    }

    // ---- 通過経路のうち、引用されたチャンクにたどり着いた経路
    var paths = citedPaths(edges, trace.chunks || [], trace.citations || []);
    var onCited = {};
    paths.edges.forEach(function (id) { onCited[id] = true; });
    edges.forEach(function (e) { e.citedPath = !!onCited[e.id]; });

    // ---- 1 ノードの上限(登場チャンク数とあふれた分)
    hops.forEach(function (hop, h) {
      var counts = (hop && hop.node_chunk_counts) || {};
      var capped = (hop && hop.capped_chunk_ids) || {};
      var hopCapped = (hop && hop.hop_capped_by_entity) || {};
      ((hop && hop.selected) || []).forEach(function (eid) {
        var node = nodeById[nodeKey(h, eid)];
        if (node) node.chunkSource = (hop && hop.chunk_source) || null;
      });
      Object.keys(counts).forEach(function (eid) {
        var node = nodeById[nodeKey(h, eid)];
        if (node) node.appearanceCount = counts[eid];
      });
      Object.keys(capped).forEach(function (eid) {
        var node = nodeById[nodeKey(h, eid)];
        if (node) node.cappedChunkIds = (capped[eid] || []).map(String);
      });
      Object.keys(hopCapped).forEach(function (eid) {
        var node = nodeById[nodeKey(h, eid)];
        if (node) node.hopCappedChunkIds = (hopCapped[eid] || []).map(String);
      });
    });

    // ---- チャンクのバッジ
    (trace.chunks || []).forEach(function (ch) {
      var main = nodeKey(ch.hop, ch.via_entity);
      addBadge(main, ch.id, ch.hop, false, ch.evidence_of);
      (ch.also_via || []).forEach(function (a) {
        addBadge(nodeKey(a.hop, a.entity), ch.id, a.hop, true);
      });
    });

    function addBadge(key, chunkId, hop, also, evidenceOf) {
      var node = nodeById[key];
      if (!node) return;
      var id = String(chunkId);
      if (node.chunks.some(function (b) { return b.id === id; })) return;
      var badge = {
        id: id, node: key, hop: hop, also: also,
        cited: !!cited[id.toLowerCase()],
        required: !!required[id.toLowerCase()],
        evidenceOf: (evidenceOf || []).slice(),
        appearStep: stepOf(hop, "chunks")
      };
      node.chunks.push(badge);
      badges.push(badge);
    }

    // ---- バッジの並び: そのノード経由で取ったもの(取得順)→ 別経路で再び出会ったもの
    nodes.forEach(function (n) {
      var primary = n.chunks.filter(function (b) { return !b.also; });
      var also = n.chunks.filter(function (b) { return b.also; });
      n.chunks = primary.concat(also);
    });

    // ---- 列の中の並び: 選ばれたもの → スコアの高い順 → 名前順
    columns.forEach(function (col) {
      var inCol = nodes.filter(function (n) { return n.col === col.index; });
      inCol.sort(function (a, b) {
        if (a.selected !== b.selected) return a.selected ? -1 : 1;
        var sa = a.score == null ? -1 : a.score;
        var sb = b.score == null ? -1 : b.score;
        if (sa !== sb) return sb - sa;
        return a.name < b.name ? -1 : a.name > b.name ? 1 : 0;
      });
      inCol.forEach(function (n, i) { n.row = i; });
      col.nodes = inCol.map(function (n) { return n.id; });
    });

    var totalSteps = columns.length * STEPS_PER_COLUMN + 1;
    var steps = [];
    columns.forEach(function (col) {
      var label = col.index === 0 ? "入口" : "ホップ" + col.index;
      steps.push(label + ": 候補を評価");
      steps.push(label + ": 選ぶ");
      steps.push(label + ": チャンクを取得");
      steps.push(label + ": 十分性");
    });
    steps.push("終了: " + (trace.stop_reason || "?"));

    return {
      columns: columns,
      nodes: nodes,
      edges: edges,
      badges: badges,
      steps: steps,
      totalSteps: totalSteps,
      stopStep: totalSteps - 1,
      citedPathNodes: paths.nodes,
      stopReason: trace.stop_reason || null,
      haltedAt: trace.halted_at || null,
      mode: trace.mode || null,
      minNeighborScore: trace.params && trace.params.min_neighbor_score != null
        ? trace.params.min_neighbor_score : null,
      maxChunksPerNode: trace.params && trace.params.max_chunks_per_node != null
        ? trace.params.max_chunks_per_node : null,
      maxChunksPerHop: trace.params && trace.params.max_chunks_per_hop != null
        ? trace.params.max_chunks_per_hop : null,
      entryFallback: entryFallback(trace),
      typeColors: typeColors(master)
    };
  }

  /**
   * 入口が 0 件のときの意味での聞き直し(trace.entry_fallback_used / entry_fallback)のまとめ。
   * foundBySemantic = 聞き直しで選ばれた入口の数(entry[].via === "semantic" かつ selected)。
   */
  function entryFallback(trace) {
    var fb = trace.entry_fallback || {};
    var found = (trace.entry || []).filter(function (e) {
      return e.via === "semantic" && e.selected;
    }).length;
    return {
      used: !!trace.entry_fallback_used,
      foundBySemantic: found,
      selectedTypes: fb.selected_types || [],
      typeScores: fb.type_scores || [],
      threshold: fb.threshold == null ? null : fb.threshold,
      nQuestions: fb.n_questions == null ? null : fb.n_questions
    };
  }

  function makeColumn(index, hop) {
    hop = hop || {};
    return {
      index: index,
      hop: index,
      sufficiency: hop.sufficiency == null ? null : hop.sufficiency,
      sufficiencyReused: !!hop.sufficiency_reused,
      newChunkIds: hop.new_chunk_ids || [],
      droppedChunkIds: hop.dropped_chunk_ids || [],
      hopCappedChunkIds: (hop.hop_capped_chunk_ids || []).map(String),
      sufficiencyStep: stepOf(index, "sufficiency"),
      nodes: []
    };
  }

  function markSelectedEdges(edges, hop, chosen) {
    var best = {};
    edges.forEach(function (e) {
      if (e.hop !== hop) return;
      var entity = e.to.split(":").slice(1).join(":");
      if (!chosen[entity]) return;
      var s = e.score == null ? -1 : e.score;
      if (!(entity in best) || s > best[entity].score) {
        best[entity] = {score: s, base: e.id.replace(/-\d+$/, "")};
      }
    });
    edges.forEach(function (e) {
      if (e.hop !== hop) return;
      var entity = e.to.split(":").slice(1).join(":");
      if (best[entity] && e.id.replace(/-\d+$/, "") === best[entity].base) e.selected = true;
    });
  }

  /**
   * 引用されたチャンクにたどり着いた経路。
   * 引用チャンク(citations の chunk_ids)を**最初に取った**ノード = chunks[] の (hop 列, via_entity) だけ。
   * also_via(後のホップで別の経路から再び出会っただけのノード)は含めない(そこを通らなくても取れていたため)。
   * そこから「選んで進んだエッジ(selected)」を入口までさかのぼる。行き先ノードで via_edge と
   * 同じ関係の選んだエッジがあればそれだけをたどり、無ければそのノードに入る選んだエッジを全部たどる。
   * さかのぼった先(前の列)のノードには関係の指定は無いので、入る選んだエッジを全部たどる。
   * @param edges     buildGraph のエッジ({id, from, to, edge, selected})
   * @param chunks    trace.chunks
   * @param citations trace.citations
   * @returns {edges: [エッジ ID], nodes: [ノード ID(引用チャンクを持つノードと、経路上のノード)]}
   */
  function citedPaths(edges, chunks, citations) {
    var cited = citedSet({citations: citations});
    var incoming = {};
    (edges || []).forEach(function (e) {
      if (!e.selected) return;
      (incoming[e.to] = incoming[e.to] || []).push(e);
    });
    var edgeOut = [], edgeSeen = {}, nodeOut = [], nodeSeen = {}, visited = {};
    function visit(key, via) {
      var vkey = key + "|" + (via || "");
      if (visited[vkey]) return;
      visited[vkey] = true;
      if (!nodeSeen[key]) { nodeSeen[key] = true; nodeOut.push(key); }
      var into = incoming[key] || [];
      if (via) {
        var same = into.filter(function (e) { return e.edge === via; });
        if (same.length) into = same;
      }
      into.forEach(function (e) {
        if (!edgeSeen[e.id]) { edgeSeen[e.id] = true; edgeOut.push(e.id); }
        visit(e.from, null);
      });
    }
    (chunks || []).forEach(function (ch) {
      if (!cited[String(ch.id).toLowerCase()]) return;
      if (ch.hop != null && ch.via_entity != null) visit(nodeKey(ch.hop, ch.via_entity), ch.via_edge || null);
    });
    return {edges: edgeOut, nodes: nodeOut};
  }

  /**
   * エッジの流れのアニメーション。選択の段階に達した「選んで進んだエッジ」だけが流れる。
   * @returns null(流さない)| {kind: "cited"|"passed", reverse: 左向き(IN)なら true。向きなし(SYM)は false}
   */
  function flowOf(edge, step) {
    if (!edge.selected || !stateAt(edge, step).visible || !stateAt(edge, step).selectedShown) return null;
    return {kind: edge.citedPath ? "cited" : "passed", reverse: edge.arrow === "left"};
  }

  /**
   * 再生の段階 step のときの見え方。
   * @returns {visible, selectedShown}  visible=まだ出ていなければ false、selectedShown=選択の強調を出してよいか
   */
  function stateAt(item, step) {
    return {
      visible: step >= item.appearStep,
      selectedShown: item.selectStep == null ? true : step >= item.selectStep
    };
  }

  // ---------------------------------------------------------------- エッジのラベル(行き先ノードの側に置く)

  /** 相手ノード名の短縮(「モンキー・D・ガープ」→「ガープ」)。max 文字(既定 6)を超えたら切って「…」。 */
  function shortName(name, max) {
    if (!name) return "";
    max = max || 6;
    var parts = String(name).split("・").filter(function (x) { return x.length >= 2; });
    var s = parts.length ? parts[parts.length - 1] : String(name);
    return s.length > max ? s.slice(0, max - 1) + "…" : s;
  }

  // 関係名の語尾(「家族・幼なじみである」→「家族・幼なじみ」、「拠点とする」→「拠点」)。長いものから順に試す。
  // v3 の粗い関係名も同じ規則で縮める(「間柄がある」→「間柄」、「組織と関わる」→「組織」、
  // 「組織どうし関わる」→「組織どうし」)。「と関わる」は「関わる」より先に試す
  var REL_SUFFIXES = ["と関わる", "関わる", "とする", "である", "される", "がある", "する"];
  /** 関係名を図のラベル用に短くする(語尾を省く。省くと 1 文字以下になるものはそのまま)。全文はホバーの詳細で見る。 */
  function shortRelation(rel) {
    var s = String(rel || "");
    for (var i = 0; i < REL_SUFFIXES.length; i++) {
      var suf = REL_SUFFIXES[i];
      if (s.length - suf.length >= 2 && s.slice(-suf.length) === suf) return s.slice(0, -suf.length);
    }
    return s;
  }

  /** 文字幅の見積もり(px)。全角は fontSize、半角は 0.6 倍。 */
  function textWidth(text, fontSize) {
    fontSize = fontSize || 10;
    var w = 0;
    for (var i = 0; i < text.length; i++) w += text.charCodeAt(i) > 0xff ? fontSize : fontSize * 0.6;
    return Math.ceil(w);
  }

  /** text を幅 maxW(px)に収める。はみ出すなら末尾を削って「…」を付ける。 */
  function clipToWidth(text, maxW, fontSize) {
    text = String(text);
    if (textWidth(text, fontSize) <= maxW) return text;
    var out = text;
    while (out.length > 1 && textWidth(out + "…", fontSize) > maxW) out = out.slice(0, -1);
    return out + "…";
  }

  // ラベルの本文(関係名 + 相手名)の最大幅(px、10px の文字で)。スコアは別に小さく後ろに付ける
  var LABEL_MAX_W = 100;
  var LABEL_PARTNER_MAX = 4;   // IN の「←相手名」の相手名は 4 文字まで
  var SYM_MARK = "⇄";          // 向きのない関係(SYM)のラベルの頭に付ける
  var SCORE_FONT = 9;          // スコアの文字の大きさ(本文は 10)

  /**
   * ラベルの部品。main = 本文(10px)、score = スコア(小さく後ろに)。
   * full=false(選ばなかったエッジ)は本文なし(IN だけ相手名)。full=true は短くした関係名
   * (語尾を省く)+ IN なら「←相手名」。本文が LABEL_MAX_W を超えるなら関係名を削って「…」。
   * IN(左向き)のラベルは矢印の先 = 左の列の側に置くので、どのエッジか分かるよう相手(右の列のノード)の
   * 名前を「←ガープ」の形で添える(短い表記でも添える)。
   * 向きのない関係(arrow="none"、SYM)は頭に「⇄」を付ける(選ばなかったエッジでも「⇄」だけは出す)。
   */
  function labelParts(edge, full, otherName) {
    var score = edge.score == null ? "—" : Number(edge.score).toFixed(2);
    var partner = edge.arrow === "left" && otherName ? "←" + shortName(otherName, LABEL_PARTNER_MAX) : "";
    var mark = edge.arrow === "none" ? SYM_MARK : "";
    if (!full) return {main: mark || partner, score: score};
    var rel = shortRelation(edge.edge);
    var room = LABEL_MAX_W - (partner ? textWidth(" " + partner) : 0) - (mark ? textWidth(mark + " ") : 0);
    rel = clipToWidth(rel, Math.max(20, room));
    return {main: (mark ? mark + " " : "") + rel + (partner ? " " + partner : ""), score: score};
  }

  /** ラベルの文字(本文 + 空白 + スコア。本文が無ければスコアだけ)。 */
  function labelText(edge, full, otherName) {
    var p = labelParts(edge, full, otherName);
    return p.main ? p.main + " " + p.score : p.score;
  }

  /** ラベルの幅の見積もり(本文は 10px、スコアは SCORE_FONT px)。 */
  function labelWidth(edge, full, otherName) {
    var p = labelParts(edge, full, otherName);
    return (p.main ? textWidth(p.main + " ", 10) : 0) + textWidth(p.score, SCORE_FONT);
  }

  /**
   * 列と列の間隔(px)。OUT のラベルは右の列の手前に右寄せ、IN のラベルは左の列の右に左寄せで置く。
   * どちらも「矢印の先から pad px + ラベルの幅 + 反対側のノードまで margin px」が入れば、ノードとは重ならない。
   * OUT と IN のラベル同士が同じ高さで重なるときは、描画後の実測(resolveOverlaps)で上へずらす。
   * そのため間隔は「OUT と IN の大きいほうの幅 + pad + margin」(合計ではない)。最低 minGap。
   */
  function columnGap(edges, opts) {
    opts = opts || {};
    var minGap = opts.minGap == null ? 140 : opts.minGap;
    var pad = opts.pad == null ? 14 : opts.pad;
    var margin = opts.margin == null ? 10 : opts.margin;
    var maxW = 0;
    var nameOf = opts.nameOf || function () { return ""; };
    (edges || []).forEach(function (e) {
      var w = labelWidth(e, !!e.selected, nameOf(e.to)) + 8;   // +8 = 背景の余白
      maxW = Math.max(maxW, w);
    });
    return Math.max(minGap, Math.ceil(maxW + pad + margin));
  }

  /**
   * 描画後の実測で重なりを解消する。boxes = [{id, group, order, x0, x1, y0, y1}](getBBox の外枠+余白)
   * group・order の順に見て、既に置いたものと重なれば上へずらす(同じ山は上へ積んでいるので上に逃がす)。
   * 戻り値は {id: dy}(負 = 上へ)。
   */
  function resolveOverlaps(boxes, gap) {
    gap = gap == null ? 2 : gap;
    var items = boxes.map(function (b) { return {id: b.id, group: b.group, order: b.order,
                                                 x0: b.x0, x1: b.x1, y0: b.y0, y1: b.y1, dy: 0}; });
    items.sort(function (a, b) {
      return a.group < b.group ? -1 : a.group > b.group ? 1 : a.order - b.order;
    });
    var placed = [];
    items.forEach(function (it) {
      var moved = true, guard = 0;
      while (moved && guard++ < 200) {
        moved = false;
        for (var i = 0; i < placed.length; i++) {
          var p = placed[i];
          var xOverlap = it.x0 < p.x1 && p.x0 < it.x1;
          var yOverlap = it.y0 + it.dy < p.y1 + p.dy + gap && p.y0 + p.dy < it.y1 + it.dy + gap;
          if (xOverlap && yOverlap) {
            it.dy = (p.y0 + p.dy) - gap - it.y1;     // p の上端の gap px 上に下端をそろえる
            moved = true;
          }
        }
      }
      placed.push(it);
    });
    var out = {};
    placed.forEach(function (p) { out[p.id] = p.dy; });
    return out;
  }

  /**
   * ラベルの置き場所。items = [{id, head(矢印の先のノード ID), headX, headY, arrow, score, selected}]
   * - arrow=right(OUT): 矢印の先端(行き先ノードの左端)の pad px 手前に**右寄せ**(anchor=end)
   * - arrow=left(IN): 矢印の先端(左の列のノードの右端)の pad px 右に**左寄せ**(anchor=start)
   * - arrow=none(SYM、向きなし): OUT と同じ置き方(右の列のノードの手前に右寄せ。同じ山に積む)
   * - 1 本目は線の少し上(headY - 5)。同じ先端に複数来るときだけ、上へ step px(既定 18 = 文字の高さ+余白)ずつ積む
   *   (描画後に画面側で getBBox() の実測で重なりを確かめ、重なっていればさらに離す)
   *   (選んだもの → スコアの高い順。1 本目が線に一番近い)
   * 戻り値は {id: {x, y, anchor}}。
   */
  function placeEdgeLabels(items, opts) {
    opts = opts || {};
    var pad = opts.pad == null ? 14 : opts.pad;
    var step = opts.step == null ? 18 : opts.step;
    var groups = {};
    (items || []).forEach(function (it) {
      var key = it.head + "|" + (it.arrow === "left" ? "left" : "right");   // none(SYM)は right と同じ山
      (groups[key] = groups[key] || []).push(it);
    });
    var out = {};
    Object.keys(groups).forEach(function (key) {
      var g = groups[key].slice();
      g.sort(function (a, b) {
        if (!!a.selected !== !!b.selected) return a.selected ? -1 : 1;
        var sa = a.score == null ? -1 : a.score, sb = b.score == null ? -1 : b.score;
        if (sa !== sb) return sb - sa;
        return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
      });
      g.forEach(function (it, k) {
        var left = it.arrow === "left";
        out[it.id] = {x: left ? it.headX + pad : it.headX - pad, y: it.headY - 5 - k * step,
                      anchor: left ? "start" : "end", group: key, order: k};
      });
    });
    return out;
  }

  // ---- 時間と費用の内訳(結果見出しのチップ用)
  function num(v) {
    return typeof v === "number" && isFinite(v) ? v : null;
  }

  /*
   * trace(query の応答)と local(ui/server.py の _local)から、Bedrock・Jev・DynamoDB の内訳を作る。
   * 費用は local.cost(今回の実費)。デモモード(local.demo)では今回は 0 なので、local.cost_est
   * (元の実行の目安)を使い costBasis を "demo_records" / "demo_tokens" / "demo_none" にする。
   * share は latency_ms.total に占める割合(0〜1)。other は合計から 3 つを引いた残り(0 未満は 0)。
   */
  function costBreakdown(trace, local) {
    trace = trace || {};
    local = local || {};
    var lat = trace.latency_ms || {};
    var tokens = trace.tokens || {};
    var demo = !!local.demo;
    var cost = demo ? (local.cost_est || {}) : (local.cost || {});
    var basis = demo ? "demo_" + ((local.cost_est && local.cost_est.basis) || "none") : "actual";
    var total = num(lat.total);
    var parts = [
      {key: "bedrock", label: "LLM(Haiku)", ms: num(lat.bedrock), usd: num(cost.gen_usd),
       inputTokens: num(tokens.bedrock_input), outputTokens: num(tokens.bedrock_output),
       calls: num(trace.answer_attempts)},
      {key: "jev", label: "Jev", ms: num(lat.jev), usd: num(cost.jev_usd),
       inputTokens: num(tokens.jev_input), outputTokens: null, calls: num(trace.jev_calls)},
      {key: "ddb", label: "DynamoDB", ms: num(lat.ddb), usd: null,
       inputTokens: null, outputTokens: null, calls: null}
    ];
    var used = 0;
    parts.forEach(function (p) {
      used += p.ms || 0;
      p.share = total && total > 0 && p.ms != null ? Math.min(p.ms / total, 1) : null;
    });
    var otherMs = total == null ? null : Math.max(total - used, 0);
    return {
      totalMs: total, parts: parts, otherMs: otherMs,
      otherShare: total && total > 0 ? otherMs / total : null,
      totalUsd: num(cost.total_usd), costBasis: basis
    };
  }

  return {
    buildGraph: buildGraph,
    costBreakdown: costBreakdown,
    entryFallback: entryFallback,
    citedPaths: citedPaths,
    flowOf: flowOf,
    labelText: labelText,
    labelParts: labelParts,
    labelWidth: labelWidth,
    shortName: shortName,
    shortRelation: shortRelation,
    clipToWidth: clipToWidth,
    SCORE_FONT: SCORE_FONT,
    SYM_MARK: SYM_MARK,
    resolveOverlaps: resolveOverlaps,
    textWidth: textWidth,
    columnGap: columnGap,
    placeEdgeLabels: placeEdgeLabels,
    stateAt: stateAt,
    typeColors: typeColors,
    nodeKey: nodeKey,
    STEPS_PER_COLUMN: STEPS_PER_COLUMN
  };
});
