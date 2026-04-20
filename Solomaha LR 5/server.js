const express = require('express');
const app = express();
const cors = require('cors');

app.use(cors());
app.use(express.json());

app.post('/square', (req, res) => {
    let number = parseFloat(req.body.number);
    let result = number * number;

    res.json({ result: result });
});

app.listen(3000, () => {
    console.log("Сервер працює на http://localhost:3000");
});