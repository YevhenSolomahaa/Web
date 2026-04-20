const express = require("express");
const path = require("path");

const app = express();
const PORT = 3000;

// підключаємо статичні файли
app.use(express.static(__dirname));

// роут
app.get("/calculator", (req, res) => {
    res.sendFile(path.join(__dirname, "index.html"));
});

// запуск сервера
app.listen(PORT, () => {
    console.log(`Server працює: http://localhost:${PORT}`);
});