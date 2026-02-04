import { useState } from "react";
import CanvasDrawComponent from "./components/canvas";
import PredictionResult from "./components/prediction";
import { predictDigit } from "./services/api";
import "./styles.css";

function App() {
  const [canvasBlob, setCanvasBlob] = useState(null);
  const [resetCanvas, setResetCanvas] = useState(false);
  const [predictionData, setPredictionData] = useState({
    prediction: null,
    confidence: null,
    probabilities: null,
    image: null,
  });

  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    if (!canvasBlob) {
      alert("Dessinez un chiffre avant de lancer la prédiction.");
      return;
    }

    setLoading(true);
    try {
      const result = await predictDigit(canvasBlob);
      setPredictionData({
        prediction: result.prediction,
        confidence: result.confidence,
        probabilities: result.probabilities,
        image: URL.createObjectURL(canvasBlob),
      });
    } catch (err) {
      alert("Erreur lors de l'appel API : " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="canvas-section">
        <h2>Dessinez un chiffre</h2>
        <CanvasDrawComponent onChange={setCanvasBlob} resetSignal={resetCanvas} />
        <div className="buttons">
          <button onClick={() => setResetCanvas(!resetCanvas)}>Effacer</button>
          <button onClick={handlePredict} disabled={loading}>
            {loading ? "Prédiction..." : "Lancer la prédiction"}
          </button>
        </div>
      </div>
      <div className="result-section">
        <PredictionResult {...predictionData} />
      </div>
    </div>
  );
}

export default App;
