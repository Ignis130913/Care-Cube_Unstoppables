const express = require('express')
const app = express()
const cors = require('cors')
const mongoose = require("mongoose");
const { log } = require('console');

mongoose               
  .connect("mongodb://localhost:27017/carecube", {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  })
  .then(() => {
    console.log("DB Connection Successfull");
  })
  .catch((err) => {
    console.log(err.message);
  });


const userSchema = new mongoose.Schema({
    username:{
      type:String
    },
    email:{
      type:String
    },
    phoneNo:{
      type:String
    },
    password:{
      type:String,
      unique:true
    },
    age:{
      type:String,
    },
    gender:{
        type:String
    },
    familyBackground:{
        type:String
    },
    currentStatus:{
        type:String,
    },
    goals:{
      type:String,
    }
  })
  const regisSchema = new mongoose.Schema({
    user:{
      type:String
    },
    cause:{
      type:String
    },
    howlong:{
      type:String
    }})

const User = mongoose.model("users", userSchema);
const Councelling = mongoose.model("councellings", regisSchema);

app.use(
    cors({
      origin: ["http://localhost:3000"],
      methods: ["GET", "POST"],
      credentials: true,
    })

  );

app.use(express.json());

app.post('/getUser',async(req,res)=>{
  console.log("inside");
  const user = await User.findOne({_id:req.body.userId});
  console.log(user);
  res.json({status:true,detail:user})
})
app.post('/login',async(req,res)=>{
    try{
        console.log("hello");
        console.log(req.body);
        const username = req.body.username;
        const password = req.body.password;
        
        const user = await User.findOne({username:username});
        if(user){
            if(user.password===password) res.json({status:true,user:user})
            else res.json({status:false,message:"Wrong password"});
        }else{
            res.json({status:false,message:"Invalid Username"});
        }
    }catch(err){
        res.json({status:false,message:"Error"});
    }
})


app.post('/register',async(req,res)=>{
  try{
      
      const username = req.body.name;
      const password = req.body.password;
      const email = req.body.email;
      const currentStatus = req.body.currentStatus;
      const age = req.body.age;
      const familyBackground = req.body.familyBackground;
      const goals = req.body.goals;
      const phoneNo = req.body.phone;
      const gender = req.body.gender;
      
      const user = await User.findOne({username:username});
      if(user){
        res.json({status:false,message:"Username Already Exist"});
      }else{
        const user = new User({
          username:username,
          email:email,
          currentStatus:currentStatus,
          age:age,
          familyBackground:familyBackground,
          goals:goals,
          phoneNo:phoneNo,
          gender:gender,
          password:password
        })
        user.save()
        .then(async(result) => {

          // Put absent for him in Student collection
          res.json({status:true,message:"User Created",user:user});
        })
        .catch((error) => {

          res.json({status:false,message:"Error"});
        })
        
      }
  }catch(err){
      res.json({status:false,message:"Error"});
  }
})



app.post('/regisCouncelling',async(req,res)=>{
  try{
    const cause = req.body.cause;
    const howlong = req.body.howlong;
    const councellings = new Councelling({
      user:"rwdw",
      cause:cause,
      howlong:howlong
    })

    councellings.save()
        .then(async(result) => {

          // Put absent for him in Student collection
          res.json({status:true,message:"success",cause:cause});
        })
        .catch((error) => {

          res.json({status:false,message:"Error"});
        })
        

  }catch(err){
    res.json({status:false,message:"Error"});
  }

})
app.listen(4000,()=>{
    console.log("server is listenning");
})
