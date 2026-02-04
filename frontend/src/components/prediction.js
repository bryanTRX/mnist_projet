import React from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

const PredictionResult = ({ prediction, confidence, probabilities, image }) => {
  const data = probabilities
    ? probabilities.map((p, i) => ({ digit: i.toString(), probability: +(p * 100).toFixed(2) }))
    : [];

  return (
    <div className="prediction-result">
      <h2>Résultat de la Prédiction</h2>
      {image && <img src={image} alt="prétraitée" className="preprocessed-img" />}
      {prediction !== null && (
        <div className="prediction-summary">
          <b>Chiffre prédit :</b> <span className="pred-digit">{prediction}</span>
          <br />
          <b>Confiance :</b> {(confidence * 100).toFixed(2)}%
        </div>
      )}
      {data.length > 0 && (
        <div className="probabilities-chart">
          <h3>Distribution des Probabilités</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
              <XAxis dataKey="digit" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="probability" fill="#1E90FF" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};

export default PredictionResult;
