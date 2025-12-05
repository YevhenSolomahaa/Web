const express = require('express');
const path = require('path');
const app = express();
const PORT = 3000;

// Middleware для статичних файлів (CSS, images тощо)
app.use(express.static(path.join(__dirname, 'public')));

// Роути для кожного методу
app.get('/float', (req, res) => {
    res.sendFile(path.join(__dirname, 'public/float.html'));
});

app.get('/flexbox', (req, res) => {
    res.sendFile(path.join(__dirname, 'public/flexbox.html'));
});

app.get('/grid', (req, res) => {
    res.sendFile(path.join(__dirname, 'public/grid.html'));
});

// Головна сторінка з посиланнями
app.get('/', (req, res) => {
    res.send(`
        <h1>Лабораторна робота №10</h1>
        <ul>
            <li><a href="/float">Float версія</a></li>
            <li><a href="/flexbox">Flexbox версія</a></li>
            <li><a href="/grid">Grid версія</a></li>
        </ul>
    `);
});

app.listen(PORT, () => {
    console.log(`Сервер запущено: http://localhost:${PORT}`);
});