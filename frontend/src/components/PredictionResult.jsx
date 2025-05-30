import React from 'react';

const PredictionResult = ({ result }) => {
    const {predicted_digit, probabilities} = result;

    const order = probabilities
    .map((confidence, index) => ({ digit: index, confidence }));
    /* .sort((a, b) => b.confidence - a.confidence) */
    
    return (
        <div className="prediction-result">
            <h2>Prediction: {predicted_digit}</h2>
            <div className="probabilities">
                {order.map(({ digit, confidence }) => (
                    <div key={digit} className="probability-bar">
                    <div className="digit-label">{digit}</div>
                    <div className="bar-container">
                        <div
                        className="bar"
                        style={{
                            width: `${Math.round(confidence * 100)}%`,
                            backgroundColor: '#00ff41'
                        }}
                        />
                    </div>
                    <div className="percentage">{(confidence * 100).toFixed(2).padStart(5, '0')}%</div>
                    </div>
                ))}
            </div>
        </div>
    ); 
};

export default PredictionResult;