const info = document.createElement("div");
document.body.appendChild(info);

info.innerHTML = `
<b>Опис роботи калькулятора:</b><br>
Даний калькулятор реалізований на мові JavaScript з використанням об'єктної моделі документа (DOM). 
Інтерфейс створюється динамічно, без використання HTML-розмітки.

Користувач може виконувати арифметичні операції (додавання, віднімання, множення, ділення, відсотки, зміна знаку) 
за допомогою кнопок на екрані.

Також реалізована підтримка клавіатури:
• цифри (0–9)<br>
• операції (+, -, *, /)<br>
• Enter або = — обчислення результату<br>
• Backspace — видалення останнього символу<br>
• Esc — повне очищення поля

Перед обчисленням виконується перевірка введеного виразу, що запобігає помилкам під час розрахунку.
`;

info.style.color = "#aaa";
info.style.fontSize = "13px";
info.style.marginBottom = "12px";
info.style.lineHeight = "1.5";
// КОНТЕЙНЕР

const container = document.createElement("div");
document.body.appendChild(container);

container.style.width = "320px";
container.style.margin = "50px auto";
container.style.padding = "20px";
container.style.borderRadius = "40px";
container.style.backgroundColor = "#000";
container.style.boxShadow = "0 0 40px rgba(0,0,0,0.8)";


// ДИСПЛЕЙ

const display = document.createElement("input");
display.type = "text";
display.placeholder = "0";
container.appendChild(display);

display.style.width = "100%";
display.style.height = "80px";
display.style.fontSize = "36px";
display.style.textAlign = "right";
display.style.backgroundColor = "black";
display.style.color = "white";
display.style.border = "none";
display.style.marginBottom = "20px";
display.style.padding = "10px";
display.style.outline = "none";



// СІТКА

const btnContainer = document.createElement("div");
container.appendChild(btnContainer);

btnContainer.style.display = "grid";
btnContainer.style.gridTemplateColumns = "repeat(4, 1fr)";
btnContainer.style.gap = "12px";
btnContainer.style.justifyItems = "center";


// КНОПКИ (як iPhone)

const buttons = [
    "AC", "±", "%", "/",
    "7", "8", "9", "*",
    "4", "5", "6", "-",
    "1", "2", "3", "+",
    "0", ".", "⌫", "="
];


// ОБЧИСЛЕННЯ (з дужками)

function calculate(expression) {
    try {
        if (!expression) return "";

        if (!/^[0-9+\-*/(). ]+$/.test(expression)) {
            return "Error";
        }

        return Function('"use strict"; return (' + expression + ')')();
    } catch {
        return "Error";
    }
}

// ОБРОБКА КНОПОК

function onButtonClick(value) {
    const operators = ["+", "-", "*", "/"];

    if (value === "AC") {
        display.value = "";

    } else if (value === "=") {
        display.value = calculate(display.value);

    } else if (value === "⌫") {
        display.value = display.value.slice(0, -1);

    } else if (value === "±") {
        let num = parseFloat(display.value);
        if (!isNaN(num)) display.value = -num;

    } else if (value === "%") {
        let num = parseFloat(display.value);
        if (!isNaN(num)) display.value = num / 100;

    } else {
        let lastChar = display.value.slice(-1);

        if (operators.includes(value) && operators.includes(lastChar)) return;

        display.value += value;
    }
}

// СТВОРЕННЯ КНОПОК

buttons.forEach(text => {
    const btn = document.createElement("button");
    btn.innerText = text;
    btnContainer.appendChild(btn);

    // базові стилі
    btn.style.width = "70px";
    btn.style.height = "70px";
    btn.style.borderRadius = "50%";
    btn.style.border = "none";
    btn.style.fontSize = "22px";
    btn.style.cursor = "pointer";
    btn.style.transition = "0.15s";

    // кольори як iPhone
    if (["/", "*", "-", "+", "="].includes(text)) {
        btn.style.backgroundColor = "#ff9500";
        btn.style.color = "white";

    } else if (["AC", "±", "%"].includes(text)) {
        btn.style.backgroundColor = "#a5a5a5";
        btn.style.color = "black";

    } else {
        btn.style.backgroundColor = "#333";
        btn.style.color = "white";
    }

    // анімація натискання
    btn.onmousedown = () => {
        btn.style.transform = "scale(0.9)";
        btn.style.opacity = "0.7";
    };

    btn.onmouseup = () => {
        btn.style.transform = "scale(1)";
        btn.style.opacity = "1";
    };

    btn.onmouseleave = () => {
        btn.style.transform = "scale(1)";
        btn.style.opacity = "1";
    };

    btn.onclick = () => onButtonClick(text);
});


// КЛАВІАТУРА

document.addEventListener("keydown", (event) => {
    const key = event.key;
    const operators = ["+", "-", "*", "/"];

    if (!isNaN(key)) display.value += key;

    if (operators.includes(key)) {
        let lastChar = display.value.slice(-1);
        if (!operators.includes(lastChar)) display.value += key;
    }

    if (key === "Enter" || key === "=") {
        event.preventDefault();
        display.value = calculate(display.value);
    }

    if (key === "Backspace") {
        display.value = display.value.slice(0, -1);
    }

    if (key === "Escape") {
        display.value = "";
    }

    if (key === ".") display.value += ".";
});