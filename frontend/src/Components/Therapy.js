import React from 'react';
import Preloader from './Preloader';
import { Link } from 'react-router-dom';
import therapy from '../assets/img/therapy.jpeg'

const Therapy = () => {
  return (
    <div style={{ textAlign: 'center' }}>
      {/* Title */}
      <h1 style={{ fontSize: '2.5rem', marginBottom: '20px' }}>ART THERAPY</h1>
      {/* Why Us Section */}
      <section style={{ marginBottom: '30px' }}>
        <h2 style={{ fontSize: '2rem' }}>WHY US??</h2>
        <br></br>
        <div style={{ display: '-ms-grid', justifyContent: 'space-around' }}>
          {/* Theme */}
          <div>
            <h3>Theme</h3>
            <p>Artworks related to<br />different interesting<br />themes and<br />contrasting art</p>
            <hr />
          </div>
          {/* Art Info */}
          <div>
            <h3>Art info</h3>
            <p>Get to know more<br />about different forms of artworks,<br />its significance and<br />history behind its origin</p>
            <hr />
          </div>
          {/* Materials */}
          <div>
            <h3>Materials</h3>
            <p>Know more about the tools,<br />medium and materials<br />used to execute<br />the artform</p>
            <hr />
          </div>
        </div>
      </section>

      {/* Image */}
      <img src={therapy} alt="" style={{width:"50vh", marginBottom:'20px'}}/>
      {/* Forms */}
      <form style={{ marginBottom: '50px' }}>
        <label>User Name</label>
        <input type="text" name="user_name" placeholder="Enter name" style={{ margin: '5px' }} /><br />
        <label>E-mail</label>
        <input type="email" name="user_email" placeholder="Enter E-mail" style={{ margin: '5px' }} /><br />
        <label>Password</label>
        <input type="password" name="user_password" placeholder="Enter password" style={{ margin: '5px' }} /><br />
        <input type="submit" value="Submit" style={{ margin: '5px' }} />
      </form>

      {/* Link */}
      <Link to="/lor" style={{ fontSize: '1.2rem', textDecoration: 'none', color: 'blue' }}>Get Started Again</Link>
    </div>
  );
}

export default Therapy;
