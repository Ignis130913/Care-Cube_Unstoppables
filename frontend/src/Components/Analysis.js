import React, { useState, useEffect } from 'react';
import './analysis.css'; // External CSS file for styling
import Home from './Home'; // Assuming Home component is in the same directory

const Analysis = () => {
  const [foodLabel, setFoodLabel] = useState('Breakfast Time:');
  const [sleepLabel, setSleepLabel] = useState('Wake Up Time:');
  const [showHome, setShowHome] = useState(false);

  useEffect(() => {
    const currentTime = new Date().getHours();

    if (currentTime >= 0 && currentTime < 12) {
      setFoodLabel('Breakfast Time:');
      setSleepLabel('Wake Up Time:');
    } else if (currentTime >= 12 && currentTime < 18) {
      setFoodLabel('Lunch Time:');
      setSleepLabel('Afternoon Nap Time:');
    } else {
      setFoodLabel('Dinner Time:');
      setSleepLabel('Bedtime:');
    }
  }, []);

  const [foodTime, setFoodTime] = useState('');
  const [sleepTime, setSleepTime] = useState('');

  const handleSubmit = () => {
    // Redirect to Home component
    setShowHome(true);
  };

  if (showHome) {
    return <lor />;
  }

  return (
    <div className="analysis-container">
      <h2>Food and Sleep Cycle Analysis</h2>
      <div className="input-container">
        <label htmlFor="food-time">{foodLabel}</label>
        <input
          type="text"
          id="food-time"
          value={foodTime}
          onChange={(e) => setFoodTime(e.target.value)}
          placeholder="Enter food time"
        />
      </div>
      <div className="input-container">
        <label htmlFor="sleep-time">{sleepLabel}</label>
        <input
          type="text"
          id="sleep-time"
          value={sleepTime}
          onChange={(e) => setSleepTime(e.target.value)}
          placeholder="Enter sleep time"
        />
      </div>
      <button className="btn btn-primary submit-button" onClick={handleSubmit}>
        Submit
      </button>
    </div>
  );
};

export default Analysis;
