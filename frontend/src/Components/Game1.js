import React, { useState, useEffect } from 'react';
import swal from 'sweetalert2';
import './Game1.css'; // Import the CSS file
import { Link } from 'react-router-dom';

const words = ['Car', 'Tree', 'Diamond', 'Ship', 'Plane', 'Rocket', 'Bomb', 'Anchor', 'Car', 'Tree', 'Diamond', 'Ship', 'Plane', 'Rocket', 'Bomb', 'Anchor'];
const colors = ['#02ccba', '#f95c3c', '#aa7ecd', '#ffa600', '#d90429', '#0047ab', '#6497b1', '#88b04b', '#02ccba', '#f95c3c', '#aa7ecd', '#ffa600', '#d90429', '#0047ab', '#6497b1', '#88b04b'];

const Game1 = () => {
  const [opened, setOpened] = useState([]);
  const [match, setMatch] = useState(0);
  const [moves, setMoves] = useState(0);
  const [timer, setTimer] = useState(null);
  const [second, setSecond] = useState(0);
  const [cards, setCards] = useState([]);

  useEffect(() => {
    initGame();
  }, []);

  useEffect(() => {
    if (match === words.length / 2) {
      endGame();
    }
  }, [match]);

  const shuffle = (array) => {
    let currentIndex = array.length, temporaryValue, randomIndex;
    while (0 !== currentIndex) {
      randomIndex = Math.floor(Math.random() * currentIndex);
      currentIndex -= 1;
      temporaryValue = array[currentIndex];
      array[currentIndex] = array[randomIndex];
      array[randomIndex] = temporaryValue;
    }
    return array;
  };

  const initGame = () => {
    const shuffledWords = shuffle(words);
    const shuffledColors = shuffle(colors);
    const newCards = shuffledWords.map((word, index) => ({
      word,
      color: shuffledColors[index],
      id: index,
      isOpened: false // added isOpened property
    }));
    setCards(newCards);
    setOpened([]);
    setMatch(0);
    setMoves(0);
    clearInterval(timer);
    setSecond(0);
  };

  const setRating = (moves) => {
    if (moves > 20) return 0;
    else if (moves > 16) return 1;
    else if (moves > 10) return 2;
    else return 3;
  };

  const endGame = () => {
    clearInterval(timer);
    const score = setRating(moves);
    swal({
      allowEscapeKey: false,
      allowOutsideClick: false,
      title: 'Congratulations! You Won!',
      text: `With ${moves} Moves and ${score} Star(s) in ${second} seconds.\n Woooooo!`,
      type: 'success',
      confirmButtonColor: '#02ccba',
      confirmButtonText: 'Play again!'
    }).then((isConfirm) => {
      if (isConfirm) {
        initGame();
      }
    });
  };

  const handleClick = (id) => {
    if (!opened.some(card => card.id === id)) {
      const newOpened = [...opened, { id, color: cards.find(card => card.id === id).color }];
      setOpened(newOpened);
      const updatedCards = cards.map(card => {
        if (card.id === id) {
          return { ...card, isOpened: true }; // set isOpened to true when clicked
        }
        return card;
      });
      setCards(updatedCards);
      if (newOpened.length === 2) {
        setMoves(moves + 1);
        if (newOpened[0].color === newOpened[1].color) {
          setMatch(match + 1);
          setOpened([]);
        } else {
          setTimeout(() => {
            setOpened([]);
            const updatedCards = cards.map(card => ({ ...card, isOpened: false }));
            setCards(updatedCards);
          }, 1000);
        }
      }
    }
  };

  const startTimer = () => {
    setTimer(setInterval(() => {
      setSecond(second => second + 1);
    }, 1000));
  };

  const handleRestart = () => {
    swal({
      allowEscapeKey: false,
      allowOutsideClick: false,
      title: 'Are you sure?',
      text: "Your progress will be lost!",
      type: 'warning',
      showCancelButton: true,
      confirmButtonColor: '#02ccba',
      cancelButtonColor: '#f95c3c',
      confirmButtonText: 'Yes, Restart Game!'
    }).then((isConfirm) => {
      if (isConfirm) {
        initGame();
      }
    });
  };

  useEffect(() => {
    if (match === 0 && moves === 0) {
      startTimer();
    }
  }, [match, moves]);

  return (
    <div className="container">
      
      <div className="score-panel">
        <ul className="stars">
          <li><i className="fa fa-star"></i></li>
          <li><i className="fa fa-star"></i></li>
          <li><i className="fa fa-star"></i></li>
        </ul>
        <span className="moves">{moves}</span> Moves | Time: <span className="timer">{second}</span> s
        <Link to='/homec'><div className="restart" onClick={handleRestart}>
          <i className="fa fa-repeat"></i>
        </div></Link>
      </div>
      <div className="deck">
        {cards.map((card) => (
          <div key={card.id} className={`card ${opened.some(o => o.id === card.id) ? 'open' : ''}`} onClick={() => handleClick(card.id)} style={{ backgroundColor: card.isOpened ? card.color : '#2e3d49' }}>
            {opened.some(o => o.id === card.id) ? <span className="emoji">{card.word}</span> : null}
          </div>
        ))}
      </div>
    </div>
  );
};

export default Game1;

