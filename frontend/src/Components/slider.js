import React, { useState, useEffect } from 'react';
import './slider.css';

const Slider = ({ slides }) => {
  const [currentSlideIndex, setCurrentSlideIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      nextSlide();
    }, 5000);
    return () => clearInterval(interval);
  }, [currentSlideIndex]);

  const nextSlide = () => {
    setCurrentSlideIndex((prevIndex) =>
      prevIndex === slides.length - 1 ? 0 : prevIndex + 1
    );
  };

  return (
    <div className="slider-container">
      {slides.map((slide, index) => (
        <div
          key={index}
          className={`slide ${index === currentSlideIndex ? 'current' : ''}`}
          style={{ backgroundImage: `url(${slide.src})` }}
        >
          <div className="caption">
            <div className="caption-heading">
              <h1>{slide.title}</h1>
            </div>
            <div className="caption-subhead">
              <span>{slide.copy}</span>
            </div>
          </div>
        </div>
      ))}
      <a
        className="left carousel-control"
        href="#"
        onClick={() =>
          setCurrentSlideIndex(
            currentSlideIndex === 0 ? slides.length - 1 : currentSlideIndex - 1
          )
        }
      >
        &lt;
      </a>
      <a
        className="right carousel-control"
        href="#"
        onClick={() =>
          setCurrentSlideIndex(
            currentSlideIndex === slides.length - 1 ? 0 : currentSlideIndex + 1
          )
        }
      >
        &gt;
      </a>
    </div>
  );
};

export default Slider;
