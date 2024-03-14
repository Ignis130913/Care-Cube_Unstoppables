import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import axios from 'axios';
import Preloader from './Preloader';
import '../assets/css/LoR.css';

const LorR = () => {
  const [tab, setTab] = useState('sign-in');
  const [loginData, setLoginData] = useState({ username: '', password: '' });
  const [registerData, setRegisterData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
    age: '',
    gender: 'male',
    familyBackground: 'rich',
    goals: '',
    currentStatus: '',
    phone: '',
  });
  const navigate = useNavigate();

  const handleTabChange = (event) => {
    setTab(event.target.value);
  };

  const handleLoginChange = (e) => {
    const { name, value } = e.target;
    setLoginData({ ...loginData, [name]: value });
  };

  const handleRegisterChange = (e) => {
    const { name, value } = e.target;
    setRegisterData({ ...registerData, [name]: value });
  };

  const handleLoginSubmit = async (event) => {
    event.preventDefault();
    try {
      const response = await axios.post('http://localhost:4000/login', loginData, { withCredentials: true });
      const res = response.data;
      if (res.status === true) {
        navigate(`/home/${res.user._id}`);
      } else {
        alert(res.message);
        setLoginData({ ...loginData, password: '' });
      }
    } catch (ex) {
      console.log(ex);
    }
  };

  const handleRegisterSubmit = async (event) => {
    event.preventDefault();
    try {
      const response = await axios.post('http://localhost:4000/register', registerData, { withCredentials: true });
      const res = response.data;
      if (res.status === true) {
        navigate('/welcome');
      } else {
        alert(res.message);
        setRegisterData({
          ...registerData,
          password: '',
          confirmPassword: '',
        });
      }
    } catch (ex) {
      console.log(ex);
    }
  };

  return (
    <div>
      <Preloader />
      <div className="login-wrap">
        <div className="login-html">
          <input
            id="tab-1"
            type="radio"
            name="tab"
            value="sign-in"
            className="sign-in"
            checked={tab === 'sign-in'}
            onChange={handleTabChange}
          />
          <label htmlFor="tab-1" className="tab">
            Sign In
          </label>
          <input
            id="tab-2"
            type="radio"
            name="tab"
            value="sign-up"
            className="sign-up"
            checked={tab === 'sign-up'}
            onChange={handleTabChange}
          />
          <label htmlFor="tab-2" className="tab">
            Sign Up
          </label>
          <div className="login-form">
            <div className={`sign-in-htm ${tab === 'sign-in' ? 'active' : ''}`}>
              <h2>Login</h2>
              <form onSubmit={handleLoginSubmit}>
                <div className="group">
                  <label htmlFor="username" className="label">
                    Username
                  </label>
                  <input
                    id="username"
                    type="text"
                    name="username"
                    className="input"
                    value={loginData.username}
                    onChange={handleLoginChange}
                  />
                </div>
                <div className="group">
                  <label htmlFor="password" className="label">
                    Password
                  </label>
                  <input
                    id="password"
                    type="password"
                    name="password"
                    className="input"
                    value={loginData.password}
                    onChange={handleLoginChange}
                  />
                </div>
                <div className="group">
                  <input type="submit" className="button" value="Login" />
                </div>
              </form>
            </div>
            <div className={`sign-up-htm ${tab === 'sign-up' ? 'active' : ''}`} style={{overflow: 'auto', height: '70vh'}}>
              <h2>Register</h2>
              <form onSubmit={handleRegisterSubmit} style={{marginRight: '2vh'}}>
                <div className="group">
                  <label htmlFor="name" className="label">
                    Name
                  </label>
                  <input
                    id="name"
                    type="text"
                    name="name"
                    className="input"
                    value={registerData.name}
                    onChange={handleRegisterChange}
                  />
                </div>
                <div className="group">
                  <label htmlFor="email" className="label">
                    Email
                  </label>
                  <input
                    id="email"
                    type="email"
                    name="email"
                    className="input"
                    value={registerData.email}
                    onChange={handleRegisterChange}
                  />
                </div>
                <div className="group">
                  <label htmlFor="password" className="label">
                    Password
                  </label>
                  <input
                    id="password"
                    type="password"
                    name="password"
                    className="input"
                    value={registerData.password}
                    onChange={handleRegisterChange}
                  />
                </div>
                <div className="group">
                  <label htmlFor="confirmPassword" className="label">
                    Confirm Password
                  </label>
                  <input
                    id="confirmPassword"
                    type="password"
                    name="confirmPassword"
                    className="input"
                    value={registerData.confirmPassword}
                    onChange={handleRegisterChange}
                  />
                </div>
                <div className="group">
                  <label htmlFor="age" className="label">
                    Age
                  </label>
                  <input
                    id="age"
                    type="number"
                    name="age"
                    className="input"
                    value={registerData.age}
                    onChange={handleRegisterChange}
                  />
                </div>
                <div className="group">
                  <label htmlFor="gender" className="label">
                    Gender
                  </label>
                  <select
                    id="gender"
                    name="gender"
                    className="input"
                    value={registerData.gender}
                    onChange={handleRegisterChange}
                  >
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                    <option value="other">Other</option>
                  </select>
                </div>
                <div className="group">
                  <label htmlFor="familyBackground" className="label">
                    Family Background
                  </label>
                  <select
                    id="familyBackground"
                    name="familyBackground"
                    className="input"
                    value={registerData.familyBackground}
                    onChange={handleRegisterChange}
                  >
                    <option value="rich">Rich</option>
                    <option value="poor">Poor</option>
                  </select>
                </div>
                <div className="group">
                  <label htmlFor="goals" className="label">
                    Goals
                  </label>
                  <textarea
                    id="goals"
                    name="goals"
                    className="input"
                    value={registerData.goals}
                    onChange={handleRegisterChange}
                  ></textarea>
                </div>
                <div className="group">
                  <label htmlFor="currentStatus" className="label">
                    Current Status
                  </label>
                  <textarea
                    id="currentStatus"
                    name="currentStatus"
                    className="input"
                    value={registerData.currentStatus}
                    onChange={handleRegisterChange}
                  ></textarea>
                </div>
                <div className="group">
                  <label htmlFor="phone" className="label">
                    Phone
                  </label>
                  <input
                    id="phone"
                    type="tel"
                    name="phone"
                    className="input"
                    value={registerData.phone}
                    onChange={handleRegisterChange}
                  />
                </div>
                <div className="group">
                  <input type="submit" className="button" value="Register" />
                </div>
              </form>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LorR;
