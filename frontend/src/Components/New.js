import React, { useState, useEffect } from 'react';
import styled, { keyframes } from 'styled-components';
import { useInView } from 'react-intersection-observer';

import webImage from '../assets/img/puzzle.jpg';
import uiuxImage from '../assets/img/puzzle.jpg';
import posterImage from '../assets/img/puzzle.jpg';
import pythonImage from '../assets/img/puzzle.jpg';
import cyberImage from '../assets/img/puzzle.jpg';
import gameImage from '../assets/img/puzzle.jpg'

const courses = [
  {
    title: 'WEB DEVELOPMENT',
    image: webImage,
    description: `
      Learn the fundamentals of web development and build modern websites.
      Explore HTML, CSS, JavaScript, and popular web frameworks.
      Get hands-on experience with real-world projects.
      Start your journey to become a skilled web developer.
      `,
  },
  {
    title: 'UI/UX',
    image: uiuxImage,
    description: `
      Explore the principles of User Interface (UI) and User Experience (UX) design.
      Learn to create intuitive and user-friendly designs.
      Dive into prototyping, wireframing, and usability testing.
      Enhance your skills in visual and interactive design.
      `,
  },
  {
    title: 'POSTER DESIGN',
    image: posterImage,
    description: `
      Unlock your creativity with poster design and visual communication skills.
      Learn the art of composition and typography.
      Express your ideas through compelling and impactful posters.
      Turn your design passion into stunning visual stories.
      `,
  },
  {
    title: 'PYTHON PROGRAMMING',
    image: pythonImage,
    description: `
      Master the Python programming language for versatile application development.
      Explore data science, machine learning with Python.
      Dive into frameworks like Django and Flask for web applications.
      Build a strong foundation in Python programming.
      `,
  },
  {
    title: 'CYBER SECURITY',
    image: cyberImage,
    description: `
      Enhance your skills in cybersecurity and protect digital systems from threats.
      Learn ethical hacking and penetration testing best practices.
      Understand network security, cryptography, and risk management.
      Become a cybersecurity professional and safeguard the digital world.
      `,
  },
  {
    title: 'GAME DEVELOPMENT',
    image: gameImage,
    description: `
    Dive into the captivating realm of game development, where creativity knows no bounds. Master the art of crafting immersive gaming experiences, explore intricate game mechanics, and bring your ideas to life. Unleash your potential as you delve into captivating graphics, and dynamic gameplay. Whether you're creating engaging narratives or designing thrilling challenges, embrace the journey of game development and build a solid foundation for your creative endeavors
      `,
  },
];

const fadeIn = keyframes`
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
`;

const slideIn = keyframes`
  from {
    transform: translateY(100%);
  }
  to {
    transform: translateY(0);
  }
`;

const Wrapper = styled.div`
  width: 90%;
  margin: auto;
  overflow: hidden;
  position: relative;
  @media (max-width: 768px) {
    width: 100%;
  }
`;

const ButtonContainer = styled.div`
  position: absolute;
  top: 50%;
  width: 100%;
  display: flex;
  justify-content: space-between;

  @media (max-width: 768px) {
    flex-direction: row;
    align-items: center;
    top: auto;
    bottom: 10%;
  }
`;

const Button = styled.button`
  background-color: rgba(0, 0, 0, 0.8);
  color: white;
  border: none;
  padding: 10px;
  font-size: 1.5rem;
  cursor: pointer;
  outline: none;
  transition: background-color 0.3s ease-in-out, scale 0.7s ease-in-out;

  &:hover {
    background-color: rgba(0, 0, 0, 0.5);
    transform: scale(1.1);
  }

  @media (max-width: 768px) {
    margin-top: 10px;
    ${(props) => props.left && 'margin-right: auto;'}
    ${(props) => props.right && 'margin-left: auto;'}
  }
`;

const CourseContainer = styled.div`
  display: flex;
  transition: transform 1s ease-in-out;
  transform: translateX(-${(props) => props.activeCourseIndex * 100}%);
  pointer-events: ${(props) => (props.popupVisible ? 'none' : 'auto')};
`;

const CourseItem = styled.div`
  flex: 0 0 100%;
  box-sizing: border-box;
  margin: 0 10px;
`;

const CourseCard = styled.div`
  border-radius: 8px;
  overflow: hidden;
  text-align: center;
  position: relative;
  box-shadow: rgb(38, 57, 77) 0px 20px 30px -10px;
`;

const CourseImage = styled.img`
  width: 100%;  // Set width to 100% for responsiveness
  max-height: 70vh;  // Set a maximum height to avoid excessive stretching

  @media (min-width: 768px) {
    width: 70vw;
    height: 70vh;
  }

  object-fit: cover;
  cursor: pointer;
`;

const ModalOverlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  animation: ${fadeIn} 0.5s ease-in-out;
`;

const ModalContent = styled.div`
  background: white;
  padding: 20px;
  max-width: 80%;
  max-height: 80%;
  overflow-y: auto;
  border-radius: 8px;
  animation: ${slideIn} 0.5s ease-in-out;
  position: relative;
`;

const CloseButton = styled.button`
  position: absolute;
  top: 10px;
  right: 10px;
  border: none;
  font-size: 7vh; /* Adjust the font size to make it larger */
  color: black;
  cursor: pointer;
  outline: none;
  transition: transform 0.3s ease-in-out, color 0.3s ease-in-out;
  background: none;

  &:hover {
    transform: scale(1.2); /* Adjust the scale factor as needed */
    color: red; /* Change the color on hover as needed */
  }
`;


const CourseTitle = styled.h3`
  font-size: 3rem;
  color: black;
  margin-bottom: 20px;

  @media (max-width: 768px) {
    font-size: 2rem;
    margin-bottom: 10px;
  }
`;

const CourseText = styled.p`
  font-size: 1.5rem;
  line-height: 1.8;
  color: black;
  @media (max-width: 768px) {
    font-size: 1rem;
  }
`;

const Carousel = ({ data }) => {
  const [activeIndex, setActiveIndex] = useState(0);
  const [showModal, setShowModal] = useState(false);
  const [imageClicked, setImageClicked] = useState(false);

  useEffect(() => {
    const intervalId = setInterval(() => {
      if (!imageClicked) {
        setActiveIndex((prevIndex) => (prevIndex + 1) % data.length);
        setShowModal(false);
      }
      setImageClicked(false);
    }, 5000);

    return () => clearInterval(intervalId);
  }, [data, imageClicked]);

  const handleImageClick = () => {
    setShowModal(true);
    setImageClicked(true);
  };

  const handleCloseModal = () => {
    setShowModal(false);
  };

  const handleNextClick = () => {
    if (showModal) return;
    setActiveIndex((prevIndex) => (prevIndex + 1) % data.length);
    setShowModal(false);
  };

  const handlePrevClick = () => {
    if (showModal) return;
    setActiveIndex((prevIndex) => (prevIndex - 1 + data.length) % data.length);
    setShowModal(false);
  };

  return (
    <Wrapper>
      <h2
        style={{
          textAlign: 'center',
          fontSize: '3.5rem',
          marginBottom: '2rem',
          background: 'linear-gradient(black, blue)',
          backgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
        }}
      >
        Featured Courses
      </h2>
      <CourseContainer activeCourseIndex={activeIndex} popupVisible={showModal}>
        {data.map((course, index) => (
          <CourseItem key={index}>
            <CourseCard>
              <CourseImage src={course.image} alt={course.title} onClick={handleImageClick} />
            </CourseCard>
          </CourseItem>
        ))}
      </CourseContainer>
      <ButtonContainer>
      <Button left onClick={handlePrevClick}>{'<'}</Button>
      <Button right onClick={handleNextClick}>{'>'}</Button>

      </ButtonContainer>

      {showModal && (
        <ModalOverlay onClick={handleCloseModal}>
          <ModalContent onClick={(e) => e.stopPropagation()}>
          <CloseButton onClick={handleCloseModal}>&times;</CloseButton>
            <CourseImage src={data[activeIndex].image} alt={data[activeIndex].title} />
            <CourseTitle>{data[activeIndex].title}</CourseTitle>
            <CourseText>{data[activeIndex].description}</CourseText>
          </ModalContent>
        </ModalOverlay>
      )}
    </Wrapper>
  );
};


const Work = () => {
  const [ref, isInView] = useInView({
    triggerOnce: true,
    threshold: 0.5,
  });

  return <Carousel ref={ref} isInView={isInView} data={courses} />;
};

export default Work;