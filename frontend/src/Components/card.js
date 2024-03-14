// Card.js
import React, { useState } from 'react';

const Card = ({ symbol, onClick }) => {
  const [isOpen, setIsOpen] = useState(false);

  const handleClick = () => {
    setIsOpen(true);
    onClick();
  };

  return (
    <li className={`card ${isOpen ? 'open show' : ''}`} onClick={handleClick}>
      <i className={`fa fa-${symbol}`}></i>
    </li>
  );
};

export default Card;
