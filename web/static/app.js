/* =================================================================
   Smart Email Reply Generator - Client Application Logic
   ================================================================= */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const emailContext = document.getElementById("emailContext");
    const charCounter = document.getElementById("charCounter");
    const sampleSituations = document.getElementById("sampleSituations");
    const toneRadios = document.querySelectorAll('input[name="tone"]');
    const backendSelector = document.getElementById("backendSelector");
    const temperatureSlider = document.getElementById("temperatureSlider");
    const tempVal = document.getElementById("tempVal");
    const maxTokensSlider = document.getElementById("maxTokensSlider");
    const tokensVal = document.getElementById("tokensVal");
    const livePromptPreview = document.getElementById("livePromptPreview");
    const generateBtn = document.getElementById("generateBtn");
    const resetInputBtn = document.getElementById("resetInputBtn");
    
    const draftReplyEditor = document.getElementById("draftReplyEditor");
    const copyBtn = document.getElementById("copyBtn");
    const clearDraftBtn = document.getElementById("clearDraftBtn");
    const loadingIndicator = document.getElementById("loadingIndicator");
    const loadingStatusText = document.getElementById("loadingStatusText");
    const metricsBar = document.getElementById("metricsBar");
    const metricBackend = document.getElementById("metricBackend");
    const metricTone = document.getElementById("metricTone");
    const metricLatency = document.getElementById("metricLatency");
    
    const comparisonTableBody = document.getElementById("comparisonTableBody");
    const tableFilter = document.getElementById("tableFilter");
    const runAllTestsBtn = document.getElementById("runAllTestsBtn");
    const toast = document.getElementById("toast");

    // Sample Scenarios Library (Answers functional requirements & sample test cases)
    const SCENARIOS = {
        formal_leave: {
            context: "Requesting leave for one day due to a family medical appointment.",
            tone: "formal"
        },
        formal_extension: {
            context: "Requesting assignment extension for the Machine Learning final project due to illness.",
            tone: "formal"
        },
        formal_job: {
            context: "Inquiring about the status of a job application for the NLP Engineer role.",
            tone: "formal"
        },
        formal_decline: {
            context: "Declining a vendor sales pitch meeting respectfully due to budget constraints.",
            tone: "formal"
        },
        formal_report: {
            context: "Submitting project status report and requesting executive review.",
            tone: "formal"
        },
        friendly_meeting: {
            context: "Unable to attend tomorrow's meeting because I have an appointment.",
            tone: "friendly"
        },
        friendly_thanks: {
            context: "Thanking a colleague for help on debugging the deployment script today.",
            tone: "friendly"
        },
        friendly_coffee: {
            context: "Inviting teammates for a casual coffee break and project catch-up.",
            tone: "friendly"
        },
        friendly_notes: {
            context: "Sharing notes from the conference and asking if they want to collaborate.",
            tone: "friendly"
        },
        friendly_promo: {
            context: "Congratulating a teammate on their work anniversary and promotion.",
            tone: "friendly"
        }
    };

    let cachedTestCases = [];

    // Helper: Toast Notification
    function showToast(message, duration = 3000) {
        toast.textContent = message;
        toast.classList.add("show");
        setTimeout(() => toast.classList.remove("show"), duration);
    }

    // Helper: Selected Tone Getter
    function getSelectedTone() {
        const checked = document.querySelector('input[name="tone"]:checked');
        return checked ? checked.value : "formal";
    }

    function setSelectedTone(toneVal) {
        toneRadios.forEach(radio => {
            radio.checked = (radio.value === toneVal.toLowerCase());
        });
        updatePromptPreview();
    }

    // Character Counter & Live Prompt Preview
    function updateCharCountAndPreview() {
        const len = emailContext.value.length;
        charCounter.textContent = `${len} / 1000`;
        charCounter.style.color = len > 1000 ? "#dc2626" : "#94a3b8";
        updatePromptPreview();
    }

    function updatePromptPreview() {
        const tone = getSelectedTone();
        const ctx = emailContext.value.trim() || "[User's email situation will appear here]";
        
        let promptText = "";
        if (tone === "formal") {
            promptText = `Task: Generate a polite and professional formal email reply based on the situation.

[Exemplar 1: Formal Leave Request]
[Exemplar 2: Formal Meeting Absence]

Situation: ${ctx}
Tone: Formal
Draft Reply:
Dear `;
        } else {
            promptText = `Task: Generate a warm and friendly email reply based on the situation.

[Exemplar 1: Friendly Colleague Thanks]
[Exemplar 2: Friendly Lunch Catch-up]

Situation: ${ctx}
Tone: Friendly
Draft Reply:
Hi `;
        }
        livePromptPreview.textContent = promptText;
    }

    // Event Listeners for Input & Controls
    emailContext.addEventListener("input", updateCharCountAndPreview);
    toneRadios.forEach(r => r.addEventListener("change", updatePromptPreview));

    temperatureSlider.addEventListener("input", (e) => {
        tempVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    maxTokensSlider.addEventListener("input", (e) => {
        tokensVal.textContent = e.target.value;
    });

    // Sample Scenarios Dropdown Change
    sampleSituations.addEventListener("change", (e) => {
        const key = e.target.value;
        if (key && SCENARIOS[key]) {
            emailContext.value = SCENARIOS[key].context;
            setSelectedTone(SCENARIOS[key].tone);
            updateCharCountAndPreview();
            showToast(`Loaded scenario: ${SCENARIOS[key].context.substring(0, 40)}...`);
        }
    });

    // Reset Input Button
    resetInputBtn.addEventListener("click", () => {
        emailContext.value = "";
        sampleSituations.value = "";
        setSelectedTone("formal");
        updateCharCountAndPreview();
        showToast("Inputs reset to default.");
    });

    // Clear Draft Button
    clearDraftBtn.addEventListener("click", () => {
        draftReplyEditor.value = "";
        copyBtn.disabled = true;
        clearDraftBtn.disabled = true;
        metricsBar.style.display = "none";
        showToast("Draft editor cleared.");
    });

    // Copy Draft Button
    copyBtn.addEventListener("click", async () => {
        const text = draftReplyEditor.value.trim();
        if (!text) return;
        try {
            await navigator.clipboard.writeText(text);
            showToast("✅ Draft reply copied to clipboard!");
        } catch (err) {
            draftReplyEditor.select();
            document.execCommand("copy");
            showToast("✅ Draft reply copied!");
        }
    });

    // Generate Draft Reply Handler
    generateBtn.addEventListener("click", async () => {
        const contextText = emailContext.value.trim();
        const tone = getSelectedTone();
        const backend = backendSelector.value;
        const temperature = parseFloat(temperatureSlider.value);
        const maxNewTokens = parseInt(maxTokensSlider.value, 10);

        // Client-side validations
        if (!contextText) {
            showToast("⚠️ Please enter an email situation or message context.");
            emailContext.focus();
            return;
        }

        if (contextText.length > 1000) {
            showToast("⚠️ Input exceeds 1000 characters. Please shorten your context.");
            return;
        }

        // UI state: Loading
        generateBtn.disabled = true;
        loadingIndicator.style.display = "block";
        loadingStatusText.textContent = `Querying ${backendSelector.options[backendSelector.selectedIndex].text}...`;
        draftReplyEditor.value = "";
        copyBtn.disabled = true;
        clearDraftBtn.disabled = true;
        metricsBar.style.display = "none";

        try {
            const resp = await fetch("/api/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    email_context: contextText,
                    tone: tone,
                    backend: backend,
                    temperature: temperature,
                    max_new_tokens: maxNewTokens
                })
            });

            const data = await resp.json();

            if (!resp.ok || data.status === "error") {
                throw new Error(data.error || "Generation request failed.");
            }

            // Populate Draft Editor
            draftReplyEditor.value = data.draft_reply;
            copyBtn.disabled = false;
            clearDraftBtn.disabled = false;

            // Update Metrics Bar
            metricBackend.textContent = data.backend || backend;
            metricTone.textContent = (data.tone || tone).toUpperCase();
            metricLatency.textContent = `${data.latency_ms || "-"} ms`;
            metricsBar.style.display = "flex";

            showToast("✨ Draft generated! Please review and edit before sending.");

        } catch (err) {
            console.error("Generation error:", err);
            showToast(`❌ Error: ${err.message}`, 5000);
            draftReplyEditor.value = `[Generation Error]: ${err.message}\n\nPlease verify that the selected backend is active and inputs are valid.`;
        } finally {
            generateBtn.disabled = false;
            loadingIndicator.style.display = "none";
        }
    });

    // Comparison Table Functions
    async function loadTestCases() {
        try {
            const resp = await fetch("/api/test_cases");
            const data = await resp.json();
            cachedTestCases = data.test_cases || [];
            renderComparisonTable(cachedTestCases);
        } catch (err) {
            console.error("Failed to load test cases:", err);
        }
    }

    function renderComparisonTable(cases) {
        comparisonTableBody.innerHTML = "";
        const filterVal = tableFilter.value;

        const filtered = cases.filter(c => {
            if (filterVal === "all") return true;
            return c.tone.toLowerCase() === filterVal.toLowerCase();
        });

        if (filtered.length === 0) {
            comparisonTableBody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding: 2rem;">No test cases found.</td></tr>`;
            return;
        }

        filtered.forEach(tc => {
            const tr = document.createElement("tr");
            const isFormal = tc.tone.toLowerCase() === "formal";
            const badgeClass = isFormal ? "badge-formal" : "badge-friendly";
            
            tr.innerHTML = `
                <td><strong>${tc.id}</strong></td>
                <td>${escapeHtml(tc.situation)}</td>
                <td><span class="${badgeClass}">${escapeHtml(tc.tone)}</span></td>
                <td><small style="color: #64748b;">${escapeHtml(tc.expected_behaviour || "-")}</small></td>
                <td>
                    <div style="max-height: 85px; overflow-y: auto; font-size: 0.8rem; white-space: pre-line; background: #f8fafc; padding: 0.4rem; border-radius: 4px;">
                        ${escapeHtml(tc.generated_reply || "Click 'Run Live Test Suite' to generate")}
                    </div>
                </td>
                <td>
                    <button class="btn-table-load" data-id="${tc.id}" title="Load into editor">
                        Use in Editor
                    </button>
                </td>
            `;

            tr.querySelector(".btn-table-load").addEventListener("click", () => {
                emailContext.value = tc.situation;
                setSelectedTone(tc.tone);
                updateCharCountAndPreview();
                if (tc.generated_reply) {
                    draftReplyEditor.value = tc.generated_reply;
                    copyBtn.disabled = false;
                    clearDraftBtn.disabled = false;
                }
                window.scrollTo({ top: 0, behavior: "smooth" });
                showToast(`Loaded test case ${tc.id} into editor`);
            });

            comparisonTableBody.appendChild(tr);
        });
    }

    tableFilter.addEventListener("change", () => renderComparisonTable(cachedTestCases));

    // Run Full Test Suite Button
    runAllTestsBtn.addEventListener("click", async () => {
        runAllTestsBtn.disabled = true;
        runAllTestsBtn.innerHTML = `<span class="spinner" style="width:14px; height:14px; display:inline-block; margin:0 4px 0 0; vertical-align:middle; border-width:2px;"></span> Running 10 Tests...`;
        showToast("Running all 10 test cases through DistilGPT2...");

        try {
            const resp = await fetch("/api/run_test_suite", { method: "POST" });
            const data = await resp.json();
            if (data.status === "success") {
                cachedTestCases = data.results;
                renderComparisonTable(cachedTestCases);
                showToast("✅ Successfully generated replies for all 10 test cases!");
            } else {
                throw new Error(data.error || "Failed to run test suite");
            }
        } catch (err) {
            console.error("Test suite execution failed:", err);
            showToast(`❌ Error: ${err.message}`, 4000);
        } finally {
            runAllTestsBtn.disabled = false;
            runAllTestsBtn.innerHTML = `<span class="btn-icon">▶️</span> Run Live Test Suite`;
        }
    });

    function escapeHtml(str) {
        if (!str) return "";
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Initial setups
    updateCharCountAndPreview();
    loadTestCases();
});
