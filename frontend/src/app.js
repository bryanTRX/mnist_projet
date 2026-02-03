import React, { useState } from "react";
import CanvasDrawComponent from "./components/canvas";
import PredictionResult from "./components/prediction";
import { predictDigit } from "./services/api";

function App() {
  const [canvasImage, setCanvasImage] = useState(null);
  const [resetCanvas, setResetCanvas] = useState(false);
  const [predictionData, setPredictionData] = useState({
    prediction: null,
    confidence: null,
    probabilities: null,
    image: null,
  });
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    if (!canvasImage) {
      alert("Dessinez un chiffre avant de lancer la prédiction.");
      return;
    }

    const res = await fetch(canvasImage);
    const blob = await res.blob();

    setLoading(true);
    try {
      const result = await predictDigit(blob);
      setPredictionData({
        prediction: result.prediction,
        confidence: result.confidence,
        probabilities: result.probabilities,
        image: canvasImage,
      });
    } catch (err) {
      alert("Erreur lors de l'appel API : " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: "flex", gap: "2rem", padding: "2rem" }}>
      <div>
        <h2>Dessinez ici</h2>
        <button onClick={() => setResetCanvas(!resetCanvas)}>Effacer le canvas</button>
        <CanvasDrawComponent onChange={setCanvasImage} resetSignal={resetCanvas} />
        <button onClick={handlePredict} disabled={loading}>
          {loading ? "Prédiction..." : "Lancer la Prédiction"}
        </button>
      </div>
      <div>
        <PredictionResult {...predictionData} />
      </div>
    </div>
  );
}

export default App;
