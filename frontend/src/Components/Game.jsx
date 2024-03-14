import React, { useState, useEffect } from 'react';
import swal from 'sweetalert2';

const symbols = ['bicycle', 'bicycle', 'leaf', 'leaf', 'cube', 'cube', 'anchor', 'anchor', 'paper-plane-o', 'paper-plane-o', 'bolt', 'bolt', 'bomb', 'bomb', 'diamond', 'diamond'];

const Card = ({ symbol, onClick }) => {
  return (
    <li className="card" onClick={onClick} style={{ backgroundColor: '#2e3d49', height: '100px', width: '150px', display: 'inline-block', margin: '0 30px 30px 0', lineHeight: '280px', fontSize: '0', color: '#ffffff', textAlign: 'center', borderRadius: '8px', verticalAlign: 'top', cursor: 'pointer', transform: 'rotateY(180deg)', transformStyle: 'preserve-3d', transition: 'transform .3s ease' }}>
      <i className={`fa fa-${symbol}`} style={{ lineHeight: '98px' }}></i>
    </li>
  );
};

const Deck = ({ cards, onCardClick }) => {
  return (
    <ul className="deck" style={{ width: '690px', margin: '0 auto', background: 'linear-gradient(160deg, #02CCBA 0%, #AA7ECD 100%)', padding: '32px', borderRadius: '10px', boxShadow: '12px 15px 20px 0px rgba(46, 61, 73, 0.5)' }}>
      {cards.map((symbol, index) => (
        <Card key={index} symbol={symbol} onClick={() => onCardClick(index)} />
      ))}
    </ul>
  );
};

const Game = () => {
  const [opened, setOpened] = useState([]);
  const [match, setMatch] = useState(0);
  const [moves, setMoves] = useState(0);
  const [clicks, setClicks] = useState(0);
  const [timer, setTimer] = useState(null);
  const [startTime, setStartTime] = useState(null);

  useEffect(() => {
    init();
  }, []);

  useEffect(() => {
    if (match === 8) {
      endGame();
    }
  }, [match]);

  const gameTimer = () => {
    setStartTime(new Date().getTime());
    setTimer(setInterval(() => {
      let now = new Date().getTime();
      let elapsed = now - startTime;
      let minutes = Math.floor((elapsed % (1000 * 60 * 60)) / (1000 * 60));
      let seconds = Math.floor((elapsed % (1000 * 60)) / 1000);
      if (seconds < 10) {
        seconds = "0" + seconds;
      }
      let currentTime = minutes + ':' + seconds;
      // Update clock
      document.querySelector(".clock").textContent = currentTime;
    }, 750));
  };

  const init = () => {
    const cards = shuffle(symbols);
    setOpened([]);
    setMatch(0);
    setMoves(0);
    setClicks(0);
    clearInterval(timer);
    document.querySelector(".clock").textContent = "0:00";
  };

  const shuffle = (array) => {
    let index = array.length, temp, randomIndex;
    while (0 !== index) {
      randomIndex = Math.floor(Math.random() * index);
      index -= 1;
      temp = array[index];
      array[index] = array[randomIndex];
      array[randomIndex] = temp;
    }
    return array;
  };

  const setRating = (moves) => {
    let score = 3;
    if (moves > 10 && moves <= 14) {
      score = 2;
    } else if (moves > 14) {
      score = 1;
    }
    return score;
  };

  const endGame = () => {
    clearInterval(timer);
    const score = setRating(moves);
    swal({
      allowEscapeKey: false,
      allowOutsideClick: false,
      title: 'Congratulations! You Won!',
      text: `With ${moves} Moves and ${score} Star(s). Woooooo!`,
      type: 'success',
      confirmButtonColor: '#02ccba',
      confirmButtonText: 'Play again!'
    }).then((isConfirm) => {
      if (isConfirm) {
        init();
      }
    });
  };

  const onCardClick = (index) => {
    setClicks(clicks + 1);
    if (clicks === 0) {
      gameTimer();
    }
    if (opened.length > 1) {
      return;
    }
    let newOpened = [...opened];
    newOpened.push(index);
    setOpened(newOpened);
    if (newOpened.length === 2) {
      if (symbols[newOpened[0]] === symbols[newOpened[1]]) {
        setTimeout(() => {
          setMatch(match + 1);
          setOpened([]);
        }, 800);
      } else {
        setTimeout(() => {
          setOpened([]);
        }, 800);
      }
      setMoves(moves + 1);
    }
  };

  return (
    <div className="containerdiv">
      <h1>Memory Game</h1>
      <div id="score-panel">
        <div className="leftdiv">
          <ul className="stars">
            <li><i className="fa fa-star"></i></li>
            <li><i className="fa fa-star"></i></li>
            <li><i className="fa fa-star"></i></li>
          </ul>
          <span className="moves">{moves}</span> Moves
        </div>
        <div className="timers">
          <article className="timer">
            <h3 className="info">Time: <span className="clock">0:00</span></h3>
          </article>
          <div className="restart" onClick={init}>
            <i className="fa fa-repeat"></i>
          </div>
        </div>
        <div className="clear"></div>
      </div>
      <Deck cards={symbols} onCardClick={onCardClick} />
    </div>
  );
};

export default Game;
