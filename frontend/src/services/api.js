const API_URL = process.env.REACT_APP_API_URL
export async function predictDigit(imageBlob) {
    const formData = new FormData();
    formData.append("file", imageBlob, "digit.png")

    const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData
    });

    if (!response.ok) throw new Error("Erreur API");
    console.log("API_URL =", process.env.REACT_APP_API_URL);
    return response.json();
}