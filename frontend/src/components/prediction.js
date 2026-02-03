import React from "react";

const PredictionResult = ({ prediction, confidence, probabilities, image }) => {
  return (
    <div>
      <h2>Résultat de la Prédiction</h2>
      {image && <img src={image} alt="prétraitée" width={140} />}
      {prediction !== null && (
        <div style={{ textAlign: "center", fontSize: "22px" }}>
          <b>Chiffre prédit :</b> <span style={{ color: "#1E90FF" }}>{prediction}</span>
          <br />
          <b>Confiance :</b> {(confidence * 100).toFixed(2)}%
        </div>
      )}
      {probabilities && (
        <div>
          <h3>Distribution des Probabilités</h3>
          <ul>
            {probabilities.map((p, i) => (
              <li key={i}>
                {i}: {(p * 100).toFixed(2)}%
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default PredictionResult;
