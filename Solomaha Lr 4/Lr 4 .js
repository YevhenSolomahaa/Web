// 1
function seconds(total) {
    return total % 60;
}

// 2
function perimeter(side, count) {
    return side * count;
}

// 3
function fizzBuzz(n) {
    for (let i = 1; i <= n; i++) {
        if (i % 15 === 0) console.log("fizzbuzz");
        else if (i % 3 === 0) console.log("fizz");
        else if (i % 5 === 0) console.log("buzz");
        else console.log(i);
    }
}

// 4
function Calculate(a, b, c) {
    return (a + b + c) / 3;
}

// 5
// через if
function isDivisible_if(n, x, y) {
    if (n % x === 0 && n % y === 0) {
        return true;
    }
    return false;
}

// тернарний оператор
function isDivisible_ternary(n, x, y) {
    return (n % x === 0 && n % y === 0) ? true : false;
}

// без if і тернарного
function isDivisible(n, x, y) {
    return n % x === 0 && n % y === 0;
}

// 6
function arrayStats(arr) {
    let max = Math.max(...arr);
    let min = Math.min(...arr);
    let sum = arr.reduce((a, b) => a + b, 0);
    let avg = sum / arr.length;
    let odd = arr.filter(x => x % 2 !== 0);

    console.log("Max:", max);
    console.log("Min:", min);
    console.log("Sum:", sum);
    console.log("Average:", avg);
    console.log("Odd numbers:", odd);
}

// 7
function matrixTask() {
    let matrix = [];

    for (let i = 0; i < 5; i++) {
        matrix[i] = [];
        for (let j = 0; j < 5; j++) {
            let value = Math.floor(Math.random() * 21) - 10; // від -10 до 10
            if (i === j) {
                if (value < 0) value = 0;
                else value = 1;
            }
            matrix[i][j] = value;
        }
    }

    console.log(matrix);
}

// 8
function Add(a, b) {
    return a + b;
}

function Sub(a, b) {
    return a - b;
}

function Mul(a, b) {
    return a * b;
}

function Div(a, b) {
    if (b === 0) {
        return "Ділення на нуль неможливе";
    }
    return a / b;
}

// 9
function analyzeNumber(n) {
    console.log(n > 0 ? "Позитивне" : "Негативне");

    let isPrime = n > 1;
    for (let i = 2; i < n; i++) {
        if (n % i === 0) {
            isPrime = false;
            break;
        }
    }
    console.log("Просте:", isPrime);

    console.log("Ділиться на 2:", n % 2 === 0);
    console.log("Ділиться на 3:", n % 3 === 0);
    console.log("Ділиться на 5:", n % 5 === 0);
    console.log("Ділиться на 6:", n % 6 === 0);
    console.log("Ділиться на 9:", n % 9 === 0);
}

// 10
function transformArray(arr) {
    return arr.reverse().map(item =>
        typeof item === "number" ? item * item : item
    );
}

// 11
function removeDuplicates(arr) {
    return [...new Set(arr)];
}