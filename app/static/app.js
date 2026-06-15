let currentRunId = null;

const $ = (selector) => document.querySelector(selector);

const stateLabel = {
  INPUT_RECEIVED: "已接收",
  EXTRACTING_REQUIREMENTS: "抽取字段",
  NEED_INFO: "需要补充信息",
  RETRIEVING_POLICY: "检索制度",
  PLANNING_TOOLS: "规划工具",
  CALLING_TOOLS: "调用工具",
  ASSESSING_RISK: "评估风险",
  GENERATING_RECOMMENDATION: "生成建议",
  WAITING_HUMAN_APPROVAL: "等待人工确认",
  COMPLETED: "已完成",
  FAILED: "失败",
};

const recommendationLabel = {
  SUBMIT_RECOMMENDED: "建议提交",
  SUBMIT_AFTER_SUPPLEMENT: "补充后提交",
  HUMAN_REVIEW_REQUIRED: "需要人工复核",
  HUMAN_APPROVED_TO_CONTINUE: "人工已允许继续",
  PAUSE_RECOMMENDED: "建议暂缓",
  REJECT_RECOMMENDED: "不建议提交",
  NEED_MORE_INFO: "需要补充信息",
  OUT_OF_SCOPE: "非采购对话",
};

async function loadDemoCases() {
  const response = await fetch("/api/demo-cases");
  const cases = await response.json();
  const row = $("#demoRow");
  row.innerHTML = "";
  cases.forEach((item) => {
    const button = document.createElement("button");
    button.textContent = item.name;
    button.addEventListener("click", () => {
      $("#messageInput").value = item.input;
      currentRunId = null;
      $("#continueBtn").disabled = true;
    });
    row.appendChild(button);
  });
}

async function submitPrecheck(useExistingRun) {
  const message = $("#messageInput").value.trim();
  if (!message) return;
  const payload = { message };
  if (useExistingRun && currentRunId) payload.run_id = currentRunId;
  const response = await fetch("/api/precheck", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const run = await response.json();
  if (!response.ok) {
    throw new Error(run.detail || "请求失败");
  }
  renderRun(run);
}

async function submitHumanReview() {
  if (!currentRunId) return;
  const payload = {
    decision: $("#reviewDecision").value,
    reviewer_role: $("#reviewRole").value || "采购专员",
    comment: $("#reviewComment").value || null,
  };
  const response = await fetch(`/api/runs/${currentRunId}/human-review`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const run = await response.json();
  if (!response.ok) {
    throw new Error(run.detail || "确认失败");
  }
  renderRun(run);
}

function renderRun(run) {
  currentRunId = run.run_id;
  $("#runStatus").textContent = `${stateLabel[run.current_state] || run.current_state} · ${run.run_id}`;
  $("#continueBtn").disabled = run.current_state !== "NEED_INFO";
  renderStatusMetrics(run);
  renderRecommendation(run);
  renderNextActions(run);
  renderPurchase(run.purchase_request);
  renderRisks(run.risk_findings || []);
  renderCitations(run.policy_citations || []);
  renderKeyToolResults(run.tool_calls || []);
  renderAuditDetails(run);
  renderHumanPanel(run);
}

function renderStatusMetrics(run) {
  const rec = run.recommendation;
  $("#metricState").textContent = stateLabel[run.current_state] || run.current_state;
  $("#metricRisk").innerHTML = rec ? riskBadge(rec.risk_level) : "-";
  $("#metricHuman").textContent = rec && rec.human_review_required ? "需要" : rec ? "不需要" : "-";
  $("#metricRun").textContent = run.run_id || "-";
}

function renderRecommendation(run) {
  const target = $("#recommendation");
  const rec = run.recommendation;
  if (!rec) {
    target.innerHTML = '<div class="empty">提交采购需求后生成预审建议</div>';
    return;
  }
  const reasons = rec.reasons.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  target.innerHTML = `
    <div class="recommendation-head">
      ${riskBadge(rec.risk_level)}
      <span class="badge">${escapeHtml(recommendationLabel[rec.recommendation_type] || rec.recommendation_type)}</span>
    </div>
    <h3>${escapeHtml(rec.summary)}</h3>
    <ul class="compact-list">${reasons || "<li>暂无补充原因</li>"}</ul>
  `;
}

function renderNextActions(run) {
  const target = $("#nextActions");
  const rec = run.recommendation;
  const missing = (run.missing_questions || []).filter((item) => item.required);
  const actions = [];
  if (rec) actions.push(...rec.required_actions);
  if (missing.length) {
    actions.unshift(...missing.map((item) => item.question_text));
  }
  target.innerHTML = actions.length
    ? `<ol class="action-list">${actions.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ol>`
    : '<div class="empty">暂无动作</div>';
}

function renderPurchase(purchase) {
  const target = $("#purchaseSummary");
  if (!purchase) {
    target.innerHTML = '<dt>状态</dt><dd class="empty">暂无采购摘要</dd>';
    return;
  }
  const rows = [
    ["部门", purchase.department],
    ["品类", purchase.purchase_category],
    ["物品", purchase.item_name],
    ["数量", purchase.quantity],
    ["金额", purchase.amount ? `${formatMoney(purchase.amount)} 元` : null],
    ["用途", purchase.purpose],
    ["供应商", purchase.vendor_name],
    ["交付", purchase.delivery_requirement],
    ["预算科目", purchase.budget_category],
    ["科目确认", purchase.budget_category_confirmed ? "已确认" : "待确认"],
  ];
  target.innerHTML = rows
    .map(([key, value]) => `<dt>${key}</dt><dd>${escapeHtml(value ?? "未识别")}</dd>`)
    .join("");
}

function renderRisks(items) {
  const target = $("#risks");
  const visibleItems = items.filter((item) => item.risk_level !== "LOW");
  target.innerHTML = visibleItems.length
    ? visibleItems
        .map(
          (item) => `
      <div class="item">
        <div class="item-title">
          ${riskBadge(item.risk_level)}
          ${escapeHtml(item.title)}
        </div>
        <p>${escapeHtml(item.description)}</p>
        <p>${escapeHtml(item.suggested_action)}</p>
      </div>`
        )
        .join("")
    : '<div class="empty">未发现需要拦截的主要风险</div>';
}

function renderCitations(items) {
  const target = $("#citations");
  const formalItems = items.filter((item) => item.supports_conclusion);
  target.innerHTML = formalItems.length
    ? formalItems
        .slice(0, 5)
        .map(
          (item) => `
      <div class="item">
        <div class="item-title">${escapeHtml(item.doc_title)} ${escapeHtml(item.section_id)}</div>
        <p>${escapeHtml(item.section_title)}</p>
        <p>${escapeHtml(item.content_excerpt)}</p>
      </div>`
        )
        .join("")
    : '<div class="empty">暂无可支撑结论的制度引用</div>';
}

function renderKeyToolResults(items) {
  const target = $("#keyToolResults");
  const visible = items.filter((item) => item.status === "SUCCESS");
  target.innerHTML = visible.length
    ? visible
        .map(
          (item) => `
      <div class="item">
        <div class="item-title">${toolLabel(item.tool_name)}</div>
        <p>${escapeHtml(item.output_summary)}</p>
      </div>`
        )
        .join("")
    : '<div class="empty">暂无关键工具结果</div>';
}

function renderAuditDetails(run) {
  renderTrace(run.trace_events || []);
  renderToolCalls(run.tool_calls || []);
  renderRagCandidates(run.policy_citations || []);
  renderRiskEvidence(run.risk_findings || [], run.policy_citations || [], run.tool_calls || []);
}

function renderTrace(items) {
  const target = $("#trace");
  target.innerHTML = items.length
    ? items
        .map(
          (item) => `
    <li>
      ${escapeHtml(item.title)}
      <span>${escapeHtml(stateLabel[item.state] || item.state)} · ${escapeHtml(item.event_type)}</span>
    </li>`
        )
        .join("")
    : '<li><span>暂无 Trace</span></li>';
}

function renderToolCalls(items) {
  const target = $("#toolCalls");
  target.innerHTML = items.length
    ? items
        .map(
          (item) => `
      <div class="item">
        <div class="item-title">${toolLabel(item.tool_name)} · ${escapeHtml(item.status)}</div>
        <p>${escapeHtml(item.call_reason)}</p>
        <p>${escapeHtml(item.output_summary)}</p>
        <pre>${escapeHtml(JSON.stringify(item.output_data || {}, null, 2))}</pre>
      </div>`
        )
        .join("")
    : '<div class="empty">暂无工具调用</div>';
}

function renderRagCandidates(items) {
  const target = $("#ragCandidates");
  target.innerHTML = items.length
    ? items
        .map(
          (item) => `
      <div class="item">
        <div class="item-title">${escapeHtml(item.doc_title)} ${escapeHtml(item.section_id)}</div>
        <p>${escapeHtml(item.retrieval_source)} · ${item.supports_conclusion ? "支撑结论" : "候选引用"} · ${escapeHtml(item.risk_type)}</p>
        <p>${escapeHtml(item.content_excerpt)}</p>
      </div>`
        )
        .join("")
    : '<div class="empty">暂无 RAG 候选</div>';
}

function renderRiskEvidence(risks, citations, tools) {
  const target = $("#riskEvidence");
  const citationMap = Object.fromEntries(citations.map((item) => [item.citation_id, `${item.doc_title} ${item.section_id}`]));
  const toolMap = Object.fromEntries(tools.map((item) => [item.tool_call_id, toolLabel(item.tool_name)]));
  target.innerHTML = risks.length
    ? risks
        .map((risk) => {
          const refs = risk.evidence_refs
            .map((ref) => citationMap[ref] || toolMap[ref] || ref)
            .map((ref) => `<li>${escapeHtml(ref)}</li>`)
            .join("");
          return `
      <div class="item">
        <div class="item-title">${riskBadge(risk.risk_level)} ${escapeHtml(risk.title)}</div>
        <p>${escapeHtml(risk.evidence_type)}</p>
        <ul class="compact-list">${refs || "<li>无证据编号</li>"}</ul>
      </div>`;
        })
        .join("")
    : '<div class="empty">暂无风险证据</div>';
}

function renderHumanPanel(run) {
  const panel = $("#humanPanel");
  if (run.current_state !== "WAITING_HUMAN_APPROVAL" || !run.human_review) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  $("#humanReason").textContent = `${run.human_review.reviewer_role} · ${run.human_review.reason}`;
  $("#reviewRole").value = run.human_review.reviewer_role || "采购专员";
}

function riskBadge(level) {
  return `<span class="badge ${String(level).toLowerCase()}">${escapeHtml(level)}</span>`;
}

function toolLabel(name) {
  const labels = {
    budget_lookup: "预算查询",
    vendor_qualification_lookup: "供应商资质查询",
    vendor_risk_lookup: "供应商风险查询",
    historical_price_lookup: "历史价格查询",
    approval_chain_lookup: "审批链查询",
  };
  return labels[name] || name;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatMoney(value) {
  return Number(value).toLocaleString("zh-CN", { maximumFractionDigits: 0 });
}

$("#submitBtn").addEventListener("click", () => submitPrecheck(false).catch(alert));
$("#continueBtn").addEventListener("click", () => submitPrecheck(true).catch(alert));
$("#reviewBtn").addEventListener("click", () => submitHumanReview().catch(alert));

loadDemoCases().catch(alert);
