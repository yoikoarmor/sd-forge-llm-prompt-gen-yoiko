(function () {
    const TAB_IDS = ["txt2img", "img2img"];

    function moveGenPromptMount(tabId) {
        const app = gradioApp();
        const mount = app.getElementById(`${tabId}_llm_prompt_gen_mount`);
        const promptContainer = app.getElementById(`${tabId}_prompt_container`);
        const promptRow = app.getElementById(`${tabId}_prompt_row`);

        if (!mount || !promptContainer || !promptRow) {
            return;
        }

        if (mount.parentElement !== promptContainer || mount.nextElementSibling !== promptRow) {
            promptContainer.insertBefore(mount, promptRow);
        }

        mount.dataset.llmPromptGenMounted = "true";
    }

    function relabelPromptAsOptional(tabId) {
        const app = gradioApp();
        const promptRow = app.getElementById(`${tabId}_prompt_row`);
        if (!promptRow) {
            return;
        }

        const label = promptRow.querySelector("label");
        if (label) {
            const target = label.querySelector("span") || label;
            const current = (target.textContent || "").trim();
            if (current === "Prompt" || current === "Prompt (Optional)") {
                target.textContent = "Prompt (Optional)";
            }
        }

        const textarea = promptRow.querySelector("textarea");
        if (textarea && !textarea.dataset.llmPromptOptionalPatched) {
            textarea.placeholder = "Optional base prompt context";
            textarea.dataset.llmPromptOptionalPatched = "true";
        }
    }

    function moveAllGenPromptMounts() {
        TAB_IDS.forEach((tabId) => {
            moveGenPromptMount(tabId);
            relabelPromptAsOptional(tabId);
        });
    }

    onUiLoaded(moveAllGenPromptMounts);
    onAfterUiUpdate(moveAllGenPromptMounts);
    onUiTabChange(moveAllGenPromptMounts);
})();
