const form = document.getElementById("captionForm");
const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const generateButton = document.getElementById("generateButton");
const status = document.getElementById("status");
const result = document.getElementById("result");
const captionText = document.getElementById("caption");

let previewUrl = null;

imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];

    result.style.display = "none";
    captionText.textContent = "";
    status.textContent = "";

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }

    if (!file) {
        preview.removeAttribute("src");
        preview.style.display = "none";
        return;
    }

    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    preview.style.display = "block";
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const file = imageInput.files[0];

    if (!file) {
        status.textContent = "Please choose an image first.";
        return;
    }

    const formData = new FormData();
    formData.append("image", file);

    generateButton.disabled = true;
    generateButton.textContent = "Generating...";
    status.textContent = "Generating caption. This may take a little while...";
    result.style.display = "none";

    try {
        const response = await fetch("/api/generate", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Caption generation failed.");
        }

        captionText.textContent = data.caption;
        result.style.display = "block";
        status.textContent = "Caption generated successfully.";
    } catch (error) {
        status.textContent = error.message || "Something went wrong. Please try again.";
    } finally {
        generateButton.disabled = false;
        generateButton.textContent = "Generate Caption";
    }
});

window.addEventListener("beforeunload", () => {
    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }
});