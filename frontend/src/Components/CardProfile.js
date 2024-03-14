import React, { Component } from 'react';

class CardProfile extends Component {
  constructor(props) {
    super(props);
    this.state = {
      index: 3,
      currentTime: '0:00',
      musicList: [
        { name: 'Lofi 1', author: 'Susan Cain', img: 'https://getwallpapers.com/wallpaper/full/d/2/7/1498104-cool-mind-relaxing-wallpapers-1920x1240-high-resolution.jpg', audio: 'https://www.bensound.com/bensound-music/bensound-buddy.mp3', duration: '2:02' },
        { name: 'Lofi 2', author: 'Christy Hall', img: 'https://image.freepik.com/free-photo/purple-landscape_1048-4769.jpg', audio: 'https://www.bensound.com//bensound-music/bensound-sunny.mp3', duration: '2:20' },
        { name: 'Lofi 3', author: 'Chbosky Stephen', img: 'https://dynamic-media-cdn.tripadvisor.com/media/photo-o/07/55/de/18/botany-bay.jpg?w=700&h=500&s=1', audio: 'https://www.bensound.com/bensound-music/bensound-energy.mp3', duration: '2:59' },
        { name: 'Lofi 4', author: 'Jodi Picoult', img: 'https://images.saymedia-content.com/.image/c_limit%2Ccs_srgb%2Cfl_progressive%2Cq_auto:eco%2Cw_700/MTc2MjkxODU3NjQ5NzcxNzEw/loneliness-lies-sleeping-a-writing-prompt-and-a-poem.jpg', audio: 'https://www.bensound.com/bensound-music/bensound-slowmotion.mp3', duration: '3:26' }
      ],
      pause: false,
    };
    this.playerRef = React.createRef();
    this.timelineRef = React.createRef();
    this.playheadRef = React.createRef();
    this.hoverPlayheadRef = React.createRef();
  }

  componentDidMount() {
    this.playerRef.current.addEventListener("timeupdate", this.timeUpdate, false);
    this.playerRef.current.addEventListener("ended", this.nextSong, false);
    this.timelineRef.current.addEventListener("click", this.changeCurrentTime, false);
    this.timelineRef.current.addEventListener("mousemove", this.hoverTimeLine, false);
    this.timelineRef.current.addEventListener("mouseout", this.resetTimeLine, false);
  }

  componentWillUnmount() {
    this.playerRef.current.removeEventListener("timeupdate", this.timeUpdate);
    this.playerRef.current.removeEventListener("ended", this.nextSong);
    this.timelineRef.current.removeEventListener("click", this.changeCurrentTime);
    this.timelineRef.current.removeEventListener("mousemove", this.hoverTimeLine);
    this.timelineRef.current.removeEventListener("mouseout", this.resetTimeLine);
  }

  changeCurrentTime = (e) => {
    const duration = this.playerRef.current.duration;
    const playheadWidth = this.timelineRef.current.offsetWidth;
    const offsetWidht = this.timelineRef.current.offsetLeft;
    const userClickWidht = e.clientX - offsetWidht;
    const userClickWidhtInPercent = (userClickWidht * 100) / playheadWidth;

    this.playheadRef.current.style.width = userClickWidhtInPercent + "%";
    this.playerRef.current.currentTime = (duration * userClickWidhtInPercent) / 100;
  }

  hoverTimeLine = (e) => {
    const duration = this.playerRef.current.duration;
    const playheadWidth = this.timelineRef.current.offsetWidth
    const offsetWidht = this.timelineRef.current.offsetLeft;
    const userClickWidht = e.clientX - offsetWidht;
    const userClickWidhtInPercent = (userClickWidht * 100) / playheadWidth;

    if (userClickWidhtInPercent <= 100) {
      this.hoverPlayheadRef.current.style.width = userClickWidhtInPercent + "%";
    }

    const time = (duration * userClickWidhtInPercent) / 100;

    if ((time >= 0) && (time <= duration)) {
      this.hoverPlayheadRef.current.dataset.content = this.formatTime(time);
    }
  }

  resetTimeLine = () => {
    this.hoverPlayheadRef.current.style.width = 0;
  }

  timeUpdate = () => {
    const duration = this.playerRef.current.duration;
    const timelineWidth = this.timelineRef.current.offsetWidth - this.playheadRef.current.offsetWidth;
    const playPercent = 100 * (this.playerRef.current.currentTime / duration);
    this.playheadRef.current.style.width = playPercent + "%";
    const currentTime = this.formatTime(parseInt(this.playerRef.current.currentTime));
    this.setState({
      currentTime
    });
  }

  formatTime = (currentTime) => {
    const minutes = Math.floor(currentTime / 60);
    let seconds = Math.floor(currentTime % 60);

    seconds = (seconds >= 10) ? seconds : "0" + seconds % 60;

    const formatTime = minutes + ":" + seconds

    return formatTime;
  }

  updatePlayer = () => {
    const { musicList, index } = this.state;
    const currentSong = musicList[index];
    const audio = new Audio(currentSong.audio);
    this.playerRef.current.load();
  }

  nextSong = () => {
    const { musicList, index, pause } = this.state;

    this.setState({
      index: (index + 1) % musicList.length
    });
    this.updatePlayer();
    if (pause) {
      this.playerRef.current.play();
    }
  };

  prevSong = () => {
    const { musicList, index, pause } = this.state;

    this.setState({
      index: (index + musicList.length - 1) % musicList.length
    });
    this.updatePlayer();
    if (pause) {
      this.playerRef.current.play();
    }
  };


  playOrPause = () => {
    const { musicList, index, pause } = this.state;
    const currentSong = musicList[index];
    const audio = new Audio(currentSong.audio);
    if (!this.state.pause) {
      this.playerRef.current.play();
    } else {
      this.playerRef.current.pause();
    }
    this.setState({
      pause: !pause
    })
  }

  clickAudio = (key) => {
    const { pause } = this.state;

    this.setState({
      index: key
    });

    this.updatePlayer();
    if (pause) {
      this.playerRef.current.play();
    }
  }


  render() {
    const { musicList, index, currentTime, pause } = this.state;
    const currentSong = musicList[index];
    return (
      <div style={{display: 'flex', flexWrap: 'wrap', marginLeft: '7vw'}}>
        <div className="card" style={{marginRight: '2vw', padding: '2vh'}}>
        <div className="current-song">
          <audio ref={this.playerRef}>
            <source src={currentSong.audio} type="audio/ogg" />
            Your browser does not support the audio element.
          </audio>
          <div className="img-wrap">
            <img src={currentSong.img} alt="Song cover" />
          </div>
          <span className="song-name">{currentSong.name}</span>
          <span className="song-autor">{currentSong.author}</span>

          <div className="time">
            <div className="current-time">{currentTime}</div>
            <div className="end-time">{currentSong.duration}</div>
          </div>

          <div ref={this.timelineRef} id="timeline">
            <div ref={this.playheadRef} id="playhead"></div>
            <div ref={this.hoverPlayheadRef} className="hover-playhead" data-content="0:00"></div>
          </div>

          <div className="controls">
            <button onClick={this.prevSong} className="prev prev-next current-btn"><i className="fas fa-backward"></i></button>

            <button onClick={this.playOrPause} className="play current-btn">
              {
                (!pause) ? <i className="fas fa-play"></i>
                  : <i className="fas fa-pause"></i>
              }
            </button>
            <button onClick={this.nextSong} className="next prev-next current-btn"><i className="fas fa-forward"></i></button>
          </div>

        </div>
        {/* <div className="play-list">
          {musicList.map((music, key = 0) =>
            <div key={key}
              onClick={() => this.clickAudio(key)}
              className={"track " +
                (index === key && !pause ? 'current-audio' : '') +
                (index === key && pause ? 'play-now' : '')}>

              <img className="track-img" src={music.img} alt="Song cover" />
              <div className="track-discr" >
                <span className="track-name" >{music.name}</span>
                <span className="track-author" >{music.author}</span>
              </div>
              <span className="track-duration" >
                {(index === key)
                  ? currentTime
                  : music.duration
                }
              </span>
            </div>
          )}
        </div> */}
      </div>
        <div className="card" style={{marginRight: '2vw', padding: '2vh'}}>
        <div className="current-song">
          <audio ref={this.playerRef}>
            <source src={currentSong.audio} type="audio/ogg" />
            Your browser does not support the audio element.
          </audio>
          <div className="img-wrap">
            <img src={currentSong.img} alt="Song cover" />
          </div>
          <span className="song-name">{currentSong.name}</span>
          <span className="song-autor">{currentSong.author}</span>

          <div className="time">
            <div className="current-time">{currentTime}</div>
            <div className="end-time">{currentSong.duration}</div>
          </div>

          <div ref={this.timelineRef} id="timeline">
            <div ref={this.playheadRef} id="playhead"></div>
            <div ref={this.hoverPlayheadRef} className="hover-playhead" data-content="0:00"></div>
          </div>

          <div className="controls">
            <button onClick={this.prevSong} className="prev prev-next current-btn"><i className="fas fa-backward"></i></button>

            <button onClick={this.playOrPause} className="play current-btn">
              {
                (!pause) ? <i className="fas fa-play"></i>
                  : <i className="fas fa-pause"></i>
              }
            </button>
            <button onClick={this.nextSong} className="next prev-next current-btn"><i className="fas fa-forward"></i></button>
          </div>

        </div>
        {/* <div className="play-list">
          {musicList.map((music, key = 0) =>
            <div key={key}
              onClick={() => this.clickAudio(key)}
              className={"track " +
                (index === key && !pause ? 'current-audio' : '') +
                (index === key && pause ? 'play-now' : '')}>

              <img className="track-img" src={music.img} alt="Song cover" />
              <div className="track-discr" >
                <span className="track-name" >{music.name}</span>
                <span className="track-author" >{music.author}</span>
              </div>
              <span className="track-duration" >
                {(index === key)
                  ? currentTime
                  : music.duration
                }
              </span>
            </div>
          )}
        </div> */}
      </div>
        <div className="card" style={{marginRight: '2vw', padding: '2vh'}}>
        <div className="current-song">
          <audio ref={this.playerRef}>
            <source src={currentSong.audio} type="audio/ogg" />
            Your browser does not support the audio element.
          </audio>
          <div className="img-wrap">
            <img src={currentSong.img} alt="Song cover" />
          </div>
          <span className="song-name">{currentSong.name}</span>
          <span className="song-autor">{currentSong.author}</span>

          <div className="time">
            <div className="current-time">{currentTime}</div>
            <div className="end-time">{currentSong.duration}</div>
          </div>

          <div ref={this.timelineRef} id="timeline">
            <div ref={this.playheadRef} id="playhead"></div>
            <div ref={this.hoverPlayheadRef} className="hover-playhead" data-content="0:00"></div>
          </div>

          <div className="controls">
            <button onClick={this.prevSong} className="prev prev-next current-btn"><i className="fas fa-backward"></i></button>

            <button onClick={this.playOrPause} className="play current-btn">
              {
                (!pause) ? <i className="fas fa-play"></i>
                  : <i className="fas fa-pause"></i>
              }
            </button>
            <button onClick={this.nextSong} className="next prev-next current-btn"><i className="fas fa-forward"></i></button>
          </div>

        </div>
        {/* <div className="play-list">
          {musicList.map((music, key = 0) =>
            <div key={key}
              onClick={() => this.clickAudio(key)}
              className={"track " +
                (index === key && !pause ? 'current-audio' : '') +
                (index === key && pause ? 'play-now' : '')}>

              <img className="track-img" src={music.img} alt="Song cover" />
              <div className="track-discr" >
                <span className="track-name" >{music.name}</span>
                <span className="track-author" >{music.author}</span>
              </div>
              <span className="track-duration" >
                {(index === key)
                  ? currentTime
                  : music.duration
                }
              </span>
            </div>
          )}
        </div> */}
      </div>
        <div className="card" style={{marginRight: '2vw', padding: '2vh'}}>
        <div className="current-song">
          <audio ref={this.playerRef}>
            <source src={currentSong.audio} type="audio/ogg" />
            Your browser does not support the audio element.
          </audio>
          <div className="img-wrap">
            <img src={currentSong.img} alt="Song cover" />
          </div>
          <span className="song-name">{currentSong.name}</span>
          <span className="song-autor">{currentSong.author}</span>

          <div className="time">
            <div className="current-time">{currentTime}</div>
            <div className="end-time">{currentSong.duration}</div>
          </div>

          <div ref={this.timelineRef} id="timeline">
            <div ref={this.playheadRef} id="playhead"></div>
            <div ref={this.hoverPlayheadRef} className="hover-playhead" data-content="0:00"></div>
          </div>

          <div className="controls">
            <button onClick={this.prevSong} className="prev prev-next current-btn"><i className="fas fa-backward"></i></button>

            <button onClick={this.playOrPause} className="play current-btn">
              {
                (!pause) ? <i className="fas fa-play"></i>
                  : <i className="fas fa-pause"></i>
              }
            </button>
            <button onClick={this.nextSong} className="next prev-next current-btn"><i className="fas fa-forward"></i></button>
          </div>

        </div>
        {/* <div className="play-list">
          {musicList.map((music, key = 0) =>
            <div key={key}
              onClick={() => this.clickAudio(key)}
              className={"track " +
                (index === key && !pause ? 'current-audio' : '') +
                (index === key && pause ? 'play-now' : '')}>

              <img className="track-img" src={music.img} alt="Song cover" />
              <div className="track-discr" >
                <span className="track-name" >{music.name}</span>
                <span className="track-author" >{music.author}</span>
              </div>
              <span className="track-duration" >
                {(index === key)
                  ? currentTime
                  : music.duration
                }
              </span>
            </div>
          )}
        </div> */}
      </div>
        <div className="card" style={{marginRight: '2vw', padding: '2vh'}}>
        <div className="current-song">
          <audio ref={this.playerRef}>
            <source src={currentSong.audio} type="audio/ogg" />
            Your browser does not support the audio element.
          </audio>
          <div className="img-wrap">
            <img src={currentSong.img} alt="Song cover" />
          </div>
          <span className="song-name">{currentSong.name}</span>
          <span className="song-autor">{currentSong.author}</span>

          <div className="time">
            <div className="current-time">{currentTime}</div>
            <div className="end-time">{currentSong.duration}</div>
          </div>

          <div ref={this.timelineRef} id="timeline">
            <div ref={this.playheadRef} id="playhead"></div>
            <div ref={this.hoverPlayheadRef} className="hover-playhead" data-content="0:00"></div>
          </div>

          <div className="controls">
            <button onClick={this.prevSong} className="prev prev-next current-btn"><i className="fas fa-backward"></i></button>

            <button onClick={this.playOrPause} className="play current-btn">
              {
                (!pause) ? <i className="fas fa-play"></i>
                  : <i className="fas fa-pause"></i>
              }
            </button>
            <button onClick={this.nextSong} className="next prev-next current-btn"><i className="fas fa-forward"></i></button>
          </div>

        </div>
        {/* <div className="play-list">
          {musicList.map((music, key = 0) =>
            <div key={key}
              onClick={() => this.clickAudio(key)}
              className={"track " +
                (index === key && !pause ? 'current-audio' : '') +
                (index === key && pause ? 'play-now' : '')}>

              <img className="track-img" src={music.img} alt="Song cover" />
              <div className="track-discr" >
                <span className="track-name" >{music.name}</span>
                <span className="track-author" >{music.author}</span>
              </div>
              <span className="track-duration" >
                {(index === key)
                  ? currentTime
                  : music.duration
                }
              </span>
            </div>
          )}
        </div> */}
      </div>
        <div className="card" style={{marginRight: '2vw', padding: '2vh'}}>
        <div className="current-song">
          <audio ref={this.playerRef}>
            <source src={currentSong.audio} type="audio/ogg" />
            Your browser does not support the audio element.
          </audio>
          <div className="img-wrap">
            <img src={currentSong.img} alt="Song cover" />
          </div>
          <span className="song-name">{currentSong.name}</span>
          <span className="song-autor">{currentSong.author}</span>

          <div className="time">
            <div className="current-time">{currentTime}</div>
            <div className="end-time">{currentSong.duration}</div>
          </div>

          <div ref={this.timelineRef} id="timeline">
            <div ref={this.playheadRef} id="playhead"></div>
            <div ref={this.hoverPlayheadRef} className="hover-playhead" data-content="0:00"></div>
          </div>

          <div className="controls">
            <button onClick={this.prevSong} className="prev prev-next current-btn"><i className="fas fa-backward"></i></button>

            <button onClick={this.playOrPause} className="play current-btn">
              {
                (!pause) ? <i className="fas fa-play"></i>
                  : <i className="fas fa-pause"></i>
              }
            </button>
            <button onClick={this.nextSong} className="next prev-next current-btn"><i className="fas fa-forward"></i></button>
          </div>

        </div>
        {/* <div className="play-list">
          {musicList.map((music, key = 0) =>
            <div key={key}
              onClick={() => this.clickAudio(key)}
              className={"track " +
                (index === key && !pause ? 'current-audio' : '') +
                (index === key && pause ? 'play-now' : '')}>

              <img className="track-img" src={music.img} alt="Song cover" />
              <div className="track-discr" >
                <span className="track-name" >{music.name}</span>
                <span className="track-author" >{music.author}</span>
              </div>
              <span className="track-duration" >
                {(index === key)
                  ? currentTime
                  : music.duration
                }
              </span>
            </div>
          )}
        </div> */}
      </div>
        <div className="card" style={{marginRight: '2vw', padding: '2vh'}}>
        <div className="current-song">
          <audio ref={this.playerRef}>
            <source src={currentSong.audio} type="audio/ogg" />
            Your browser does not support the audio element.
          </audio>
          <div className="img-wrap">
            <img src={currentSong.img} alt="Song cover" />
          </div>
          <span className="song-name">{currentSong.name}</span>
          <span className="song-autor">{currentSong.author}</span>

          <div className="time">
            <div className="current-time">{currentTime}</div>
            <div className="end-time">{currentSong.duration}</div>
          </div>

          <div ref={this.timelineRef} id="timeline">
            <div ref={this.playheadRef} id="playhead"></div>
            <div ref={this.hoverPlayheadRef} className="hover-playhead" data-content="0:00"></div>
          </div>

          <div className="controls">
            <button onClick={this.prevSong} className="prev prev-next current-btn"><i className="fas fa-backward"></i></button>

            <button onClick={this.playOrPause} className="play current-btn">
              {
                (!pause) ? <i className="fas fa-play"></i>
                  : <i className="fas fa-pause"></i>
              }
            </button>
            <button onClick={this.nextSong} className="next prev-next current-btn"><i className="fas fa-forward"></i></button>
          </div>

        </div>
        {/* <div className="play-list">
          {musicList.map((music, key = 0) =>
            <div key={key}
              onClick={() => this.clickAudio(key)}
              className={"track " +
                (index === key && !pause ? 'current-audio' : '') +
                (index === key && pause ? 'play-now' : '')}>

              <img className="track-img" src={music.img} alt="Song cover" />
              <div className="track-discr" >
                <span className="track-name" >{music.name}</span>
                <span className="track-author" >{music.author}</span>
              </div>
              <span className="track-duration" >
                {(index === key)
                  ? currentTime
                  : music.duration
                }
              </span>
            </div>
          )}
        </div> */}
      </div>
        <div className="card" style={{marginRight: '2vw', padding: '2vh'}}>
        <div className="current-song">
          <audio ref={this.playerRef}>
            <source src={currentSong.audio} type="audio/ogg" />
            Your browser does not support the audio element.
          </audio>
          <div className="img-wrap">
            <img src={currentSong.img} alt="Song cover" />
          </div>
          <span className="song-name">{currentSong.name}</span>
          <span className="song-autor">{currentSong.author}</span>

          <div className="time">
            <div className="current-time">{currentTime}</div>
            <div className="end-time">{currentSong.duration}</div>
          </div>

          <div ref={this.timelineRef} id="timeline">
            <div ref={this.playheadRef} id="playhead"></div>
            <div ref={this.hoverPlayheadRef} className="hover-playhead" data-content="0:00"></div>
          </div>

          <div className="controls">
            <button onClick={this.prevSong} className="prev prev-next current-btn"><i className="fas fa-backward"></i></button>

            <button onClick={this.playOrPause} className="play current-btn">
              {
                (!pause) ? <i className="fas fa-play"></i>
                  : <i className="fas fa-pause"></i>
              }
            </button>
            <button onClick={this.nextSong} className="next prev-next current-btn"><i className="fas fa-forward"></i></button>
          </div>

        </div>
        {/* <div className="play-list">
          {musicList.map((music, key = 0) =>
            <div key={key}
              onClick={() => this.clickAudio(key)}
              className={"track " +
                (index === key && !pause ? 'current-audio' : '') +
                (index === key && pause ? 'play-now' : '')}>

              <img className="track-img" src={music.img} alt="Song cover" />
              <div className="track-discr" >
                <span className="track-name" >{music.name}</span>
                <span className="track-author" >{music.author}</span>
              </div>
              <span className="track-duration" >
                {(index === key)
                  ? currentTime
                  : music.duration
                }
              </span>
            </div>
          )}
        </div> */}
      </div>
      </div>
    )
  }
}

export default CardProfile;
