import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import axios from 'axios';
import Preloader from './Preloader';
import './RegsiCouncelling.css'; // Importing the CSS file

const RegisCouncelling = () => {
    const navigate = useNavigate();
    const [values, setValues] = useState({ cause: '', howlong: '' });
    const handleSubmit = async (event) => {
        event.preventDefault();
        try {
            const response = await axios.post(
                'http://localhost:4000/regisCouncelling',
                {
                    ...values,
                },
                { withCredentials: true }
            );
            const res = response.data;
            if (res.status === true) {
                navigate('/home');
            } else {
                alert(res.message);
                setValues({ cause: '', howlong: '' });
            }
        } catch (ex) {
            console.log(ex);
        }
    };
    return (
        <div className="container">
            <div className="content">
                <h2>Register for Counseling</h2>
                <form onSubmit={(e) => handleSubmit(e)}>
                    <div className="input-group">
                        <label htmlFor="cause">Cause</label>
                        <textarea
                            type="cause"
                            name="cause"
                            onChange={(e) =>
                                setValues({ ...values, [e.target.name]: e.target.value })
                            }
                        />
                    </div>
                    <div className="input-group">
                        <label htmlFor="howlong">How Long?</label>
                        <textarea
                            type="howlong"
                            name="howlong"
                            onChange={(e) =>
                                setValues({ ...values, [e.target.name]: e.target.value })
                            }
                        />
                    </div>
                    <button type="submit">Submit</button>
                </form>
            </div>
        </div>
    );
};

export default RegisCouncelling;
