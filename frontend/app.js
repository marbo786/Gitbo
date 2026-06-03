document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("analyze-form");
    const submitBtn = document.getElementById("submit-btn");
    const btnText = submitBtn.querySelector(".btn-text");
    const btnLoader = submitBtn.querySelector(".btn-loader");
    const pipelineStatus = document.getElementById("pipeline-status");
    
    const prBanner = document.getElementById("pr-banner");
    const prLink = document.getElementById("pr-link");
    const errorBanner = document.getElementById("error-banner");
    const errorList = document.getElementById("error-list");
    
    // Cards elements
    const cardIssue = document.querySelector("#card-issue-analysis");
    const cardCode = document.querySelector("#card-relevant-code");
    const cardImpl = document.querySelector("#card-implementation");
    const cardDiff = document.querySelector("#card-git-diff");
    
    let progressInterval = null;
    let progressTimer = 0;
    
    function resetUI() {
        prBanner.classList.add("hidden");
        errorBanner.classList.add("hidden");
        errorList.innerHTML = "";
        
        // Reset placeholders
        hideCardContent(cardIssue);
        hideCardContent(cardCode);
        hideCardContent(cardImpl);
        hideCardContent(cardDiff);
        
        // Reset steps
        document.querySelectorAll(".step").forEach(step => {
            step.className = "step";
        });
    }
    
    function showCardContent(cardElement, contentSelector) {
        cardElement.querySelector(".placeholder-text").classList.add("hidden");
        cardElement.querySelector(contentSelector).classList.remove("hidden");
    }
    
    function hideCardContent(cardElement) {
        cardElement.querySelector(".placeholder-text").classList.remove("hidden");
        const resultsEl = cardElement.querySelector(".analysis-results, .code-snippets-container, .implementation-results, .terminal-container");
        if (resultsEl) resultsEl.classList.add("hidden");
    }
    
    function startProgressSimulation() {
        pipelineStatus.classList.remove("hidden");
        progressTimer = 0;
        
        const step1 = document.getElementById("step-1");
        const step2 = document.getElementById("step-2");
        const step3 = document.getElementById("step-3");
        const step4 = document.getElementById("step-4");
        
        step1.className = "step active";
        
        progressInterval = setInterval(() => {
            progressTimer += 1;
            
            // Artificial delay milestones mapping to nodes
            if (progressTimer === 4) {
                step1.className = "step completed";
                step2.className = "step active";
            } else if (progressTimer === 18) {
                step2.className = "step completed";
                step3.className = "step active";
            } else if (progressTimer === 28) {
                step3.className = "step completed";
                step4.className = "step active";
            }
        }, 1000);
    }
    
    function stopProgressSimulation(success = true) {
        clearInterval(progressInterval);
        document.querySelectorAll(".step").forEach(step => {
            if (success) {
                step.className = "step completed";
            } else {
                if (step.classList.contains("active")) {
                    step.className = "step";
                }
            }
        });
    }
    
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        resetUI();
        
        const repo_url = document.getElementById("repo_url").value.trim();
        const github_token = document.getElementById("github_token").value.trim();
        const issue_description = document.getElementById("issue_description").value.trim();
        
        // Disable submit button and show spinner
        submitBtn.disabled = true;
        btnText.classList.add("hidden");
        btnLoader.classList.remove("hidden");
        
        startProgressSimulation();
        
        try {
            const response = await fetch("/api/analyze", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    repo_url,
                    github_token,
                    issue_description
                })
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.detail || "Pipeline failed during backend execution.");
            }
            
            // Check for run errors in state
            if (data.errors && data.errors.length > 0) {
                errorBanner.classList.remove("hidden");
                data.errors.forEach(err => {
                    const li = document.createElement("li");
                    li.textContent = err;
                    errorList.appendChild(li);
                });
            }
            
            const isSuccess = data.pr_result && data.pr_result.success;
            stopProgressSimulation(isSuccess);
            
            if (isSuccess) {
                // Show Success Banner
                prBanner.classList.remove("hidden");
                prLink.href = data.pr_result.pr_url;
                prLink.textContent = "View Pull Request on GitHub";
            }
            
            // 1. Render Issue Analysis
            if (data.issue_analysis && Object.keys(data.issue_analysis).length > 0) {
                showCardContent(cardIssue, ".analysis-results");
                document.getElementById("analysis-category").textContent = data.issue_analysis.category || "N/A";
                document.getElementById("analysis-difficulty").textContent = data.issue_analysis.difficulty || "N/A";
                document.getElementById("analysis-problem").textContent = data.issue_analysis.problem || "N/A";
                
                // Render skills tags
                const skillsContainer = document.getElementById("analysis-skills");
                skillsContainer.innerHTML = "";
                const skills = data.issue_analysis.skills || [];
                skills.forEach(skill => {
                    const tag = document.createElement("span");
                    tag.className = "tag";
                    tag.textContent = skill;
                    skillsContainer.appendChild(tag);
                });
            }
            
            // 2. Render Code Snippets
            if (data.retrieved_context && data.retrieved_context.length > 0) {
                showCardContent(cardCode, ".code-snippets-container");
                const snippetsEl = document.getElementById("code-snippets");
                snippetsEl.innerHTML = "";
                
                data.retrieved_context.forEach(chunk => {
                    const snippetDiv = document.createElement("div");
                    snippetDiv.className = "code-snippet";
                    
                    snippetDiv.innerHTML = `
                        <div class="snippet-file">
                            <span>${chunk.file_path}</span>
                            <span class="snippet-lines">Lines ${chunk.start_line}-${chunk.end_line}</span>
                        </div>
                        <pre><code>${escapeHTML(chunk.content)}</code></pre>
                    `;
                    snippetsEl.appendChild(snippetDiv);
                });
            }
            
            // 3. Render Implementation Plan
            if (data.implementation_plan && Object.keys(data.implementation_plan).length > 0) {
                showCardContent(cardImpl, ".implementation-results");
                document.getElementById("impl-root-cause").textContent = data.implementation_plan.root_cause || "N/A";
                document.getElementById("impl-recommended").textContent = data.implementation_plan.recommended_changes || "N/A";
                document.getElementById("impl-effort").textContent = data.implementation_plan.estimated_effort || "N/A";
            }
            
            // 4. Render Git Diff
            if (data.git_diff) {
                showCardContent(cardDiff, ".terminal-container");
                const diffEl = document.getElementById("git-diff-content");
                diffEl.innerHTML = renderDiff(data.git_diff);
            }
            
        } catch (err) {
            console.error(err);
            stopProgressSimulation(false);
            errorBanner.classList.remove("hidden");
            const li = document.createElement("li");
            li.textContent = err.message || "An unexpected network error occurred.";
            errorList.appendChild(li);
        } finally {
            // Re-enable form
            submitBtn.disabled = false;
            btnText.classList.remove("hidden");
            btnLoader.classList.add("hidden");
        }
    });
    
    function escapeHTML(text) {
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
    
    function renderDiff(diffText) {
        const lines = diffText.split("\n");
        return lines.map(line => {
            const escaped = escapeHTML(line);
            if (line.startsWith("+") && !line.startsWith("+++")) {
                return `<span class="diff-added">${escaped}</span>`;
            } else if (line.startsWith("-") && !line.startsWith("---")) {
                return `<span class="diff-removed">${escaped}</span>`;
            } else if (line.startsWith("diff") || line.startsWith("@@") || line.startsWith("index ") || line.startsWith("---") || line.startsWith("+++")) {
                return `<span class="diff-header">${escaped}</span>`;
            }
            return escaped;
        }).join("\n");
    }
});
