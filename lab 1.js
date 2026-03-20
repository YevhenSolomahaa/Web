// ===== 1. seconds =====
function seconds(total) {
    return total % 60;
}
console.log("1:", seconds(125));

// ===== 2. perimeter =====
function perimeter(side, count) {
    return side * count;
}
console.log("2:", perimeter(5, 4));

// ===== 3. fizzBuzz =====
function fizzBuzz(n) {
    console.log("3:");
    for (let i = 1; i <= n; i++) {
        if (i % 3 === 0 && i % 5 === 0) console.log("fizzbuzz");
        else if (i % 3 === 0) console.log("fizz");
        else if (i % 5 === 0) console.log("buzz");
        else console.log(i);
    }
}
fizzBuzz(15);

// ===== 4. Calculate =====
function Calculate(a, b, c) {
    return (a + b + c) / 3;
}
console.log("4:", Calculate(3, 6, 9));

// ===== 5. isDivisible =====
function isDivisible1(n, x, y) {
    return n % x === 0 && n % y === 0;
}

const isDivisible2 = (n, x, y) => (n % x === 0 && n % y === 0 ? true : false);

const isDivisible3 = (n, x, y) => !(n % x || n % y);

console.log("5:", isDivisible1(12, 3, 4));
console.log("5:", isDivisible2(12, 3, 4));
console.log("5:", isDivisible3(12, 3, 4));

// ===== 6. Масив =====
let arr = [3, 7, 1, 9, 4, 6];

let max = Math.max(...arr);
let min = Math.min(...arr);
let sum = arr.reduce((a, b) => a + b, 0);
let avg = sum / arr.length;
let odd = arr.filter(x => x % 2 !== 0);

console.log("6:");
console.log("Max:", max);
console.log("Min:", min);
console.log("Sum:", sum);
console.log("Avg:", avg);
console.log("Odd:", odd);

// ===== 7. Матриця 5x5 =====
let matrix = [];

for (let i = 0; i < 5; i++) {
    matrix[i] = [];
    for (let j = 0; j < 5; j++) {
        matrix[i][j] = Math.floor(Math.random() * 21) - 10;

        if (i === j) {
            matrix[i][j] = matrix[i][j] < 0 ? 0 : 1;
        }
    }
}

console.log("7:");
console.table(matrix);

// ===== 8. Калькулятор =====
function Add(a, b) { return a + b; }
function Sub(a, b) { return a - b; }
function Mul(a, b) { return a * b; }
function Div(a, b) { return b !== 0 ? a / b : "Помилка: ділення на 0"; }

let a = 10, b = 2;
let op = "+"; // змінюй тут

console.log("8:");
switch(op) {
    case "+": console.log(Add(a,b)); break;
    case "-": console.log(Sub(a,b)); break;
    case "*": console.log(Mul(a,b)); break;
    case "/": console.log(Div(a,b)); break;
}

// ===== 9. Аналіз числа =====
function analyzeNumber(n) {
    console.log("9:");
    console.log(n > 0 ? "Позитивне" : "Негативне");

    let isPrime = n > 1;
    for (let i = 2; i < n; i++) {
        if (n % i === 0) isPrime = false;
    }
    console.log("Просте:", isPrime);

    [2,3,5,6,9].forEach(d => {
        console.log(`Ділиться на ${d}:`, n % d === 0);
    });
}
analyzeNumber(15);

// ===== 10. Масив трансформація =====
function transformArray(arr) {
    return arr.reverse().map(x => 
        typeof x === "number" ? x * x : x
    );
}
console.log("10:", transformArray([1, "a", 3, 2]));

// ===== 11. Видалення дублікатів =====
function removeDuplicates(arr) {
    return [...new Set(arr)];
}
console.log("11:", removeDuplicates([1,2,2,4,5,4,7,8,7,3,6]));