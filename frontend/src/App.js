import React, { useState } from 'react';
import axios from 'axios';
import FormInputs from './components/FormInputs';
import ForecastResults from './components/ForecastResults';
import './index.css';
import logo from './assets/logo.jpg'; // Importing the logo

const App = () => {
  const [formData, setFormData] = useState({
    new_base_price: '',
    new_total_price: '',
    new_is_featured_sku: false,
    new_is_display_sku: false,
    new_sku_id: ''
  });
  const [prediction, setPrediction] = useState(null);
  const [justification, setJustification] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await axios.post('http://127.0.0.1:8000/predict', formData);
      setPrediction(response.data.prediction);
      setJustification(response.data.justification);
    } catch (error) {
      console.error('Error fetching prediction:', error);
      setJustification('An error occurred while fetching the prediction.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-wrapper">
      <div className="container">
        <img src={logo} alt="Logo" className="logo" />
        <h1 style={{ marginBottom: '20px' }}>Demand Forecasting Application</h1>
        <form onSubmit={handleSubmit}>
          <FormInputs formData={formData} handleChange={handleChange} />
          <button
            type="submit"
            disabled={loading}
            className="button"
          >
            {loading ? 'Processing...' : 'Generate Forecast'}
          </button>
        </form>
        <ForecastResults
          loading={loading}
          prediction={prediction}
          justification={justification}
        />
      </div>
      <footer className="footer">
        © 2025 Team Divergent
      </footer>
    </div>
  );
};

export default App;