import React from 'react';

const ForecastResults = ({ loading, prediction, justification }) => {
  if (loading) {
    return <div className="results">Processing your request...</div>;
  }

  if (prediction !== null) {
    return (
      <div className="results">
        <h3>Forecast Results:</h3>
        <p><strong>Predicted Demand:</strong> {prediction} units</p>
        <p><strong>Justification:</strong> {justification}</p>
      </div>
    );
  }

  return null;
};

export default ForecastResults;