let cursorPosition = 0;

async function setCursorPosition(value) {
	cursorPosition = value
}

async function getFibonacciMethodResult() {
	document.getElementById("loader").style.display='flex';
	let save_formula = document.getElementById('equation_showcase').cloneNode(true);
	save_formula.id = '';
	save_formula.className = '';
    let equation = document.getElementById('equation').value;
	let a = parseInt(document.getElementById('a_value').value);
	let b = parseInt(document.getElementById('b_value').value);
	let delta = parseInt(document.getElementById('delta_value').value);
	if (equation === '' || isNaN(a) || isNaN(b)){
        return;
    }
	delta = isNaN(delta) ? 0.01 : delta;
	console.log(equation, a, b, delta)

    let output = document.getElementById("output");
    let result = document.createElement('div');
    result.className = 'result';
	let calculations;
	try {
		calculations = await eel.get_fibonacci_method_result(equation, a, b, delta)();
		document.getElementById("loader").style.display='none';
	} catch (e) {
		document.getElementById("loader").style.display='none';
		result.innerHTML += "Ошибка в входных данных";
		output.prepend(result);
		return
	}

	result.innerHTML += "Исходная функция:"
	result.appendChild(document.createElement('br'))
	result.appendChild(document.createElement('br'))
	let y_label = document.createElement('mi')
	y_label.innerHTML = 'y'
	let left_bracket = document.createElement('mo')
	left_bracket.innerHTML = '('
	let y_inner_label = document.createElement('mi')
	y_inner_label.innerHTML = 'x'
	let right_bracket = document.createElement('mo')
	right_bracket.innerHTML = ')'
	let equate = document.createElement('mo')
	equate.innerHTML = '='
	save_formula.prepend(equate)
	save_formula.prepend(right_bracket)
	save_formula.prepend(y_inner_label)
	save_formula.prepend(left_bracket)
	save_formula.prepend(y_label)
	result.appendChild(save_formula)
	result.appendChild(document.createElement('br'))
	result.appendChild(document.createElement('br'))
	calculations.map(answer => {
		result.innerHTML += "Точность: "
		result.innerHTML += answer[0]
		result.appendChild(document.createElement('br'))
		result.innerHTML += "X = "
		result.innerHTML += answer[1].toFixed(6)
		result.innerHTML += " ± "
		result.innerHTML += answer[3].toFixed(6)
		result.appendChild(document.createElement('br'))
		result.innerHTML += "Количество итераций:"
		result.innerHTML += answer[4]
		result.appendChild(document.createElement('br'))
		let img = new Image();
		img.onload = function(){
		  // execute drawImage statements here
		};
		img.src = answer[5];
		result.appendChild(img);
		result.appendChild(document.createElement('br'))
	});

	output.prepend(result);
}

async function getGoldenRatioMethodResult() {
	document.getElementById("loader").style.display='flex';
	let save_formula = document.getElementById('equation_showcase').cloneNode(true);
	save_formula.id = '';
	save_formula.className = '';
    let equation = document.getElementById('equation').value;
	let a = parseInt(document.getElementById('a_value').value);
	let b = parseInt(document.getElementById('b_value').value);
	if (equation === '' || isNaN(a) || isNaN(b)){
        return;
    }

    let output = document.getElementById("output");
    let result = document.createElement('div');
    result.className = 'result';
	let calculations;
	try {
		calculations = await eel.get_golden_ratio_method_result(equation, a, b)();
		document.getElementById("loader").style.display='none';
	} catch (e) {
		document.getElementById("loader").style.display='none';
		result.innerHTML += "Ошибка в входных данных";
		output.prepend(result);
		return
	}
	result.innerHTML += "Исходная функция:"
	result.appendChild(document.createElement('br'))
	result.appendChild(document.createElement('br'))
	let y_label = document.createElement('mi')
	y_label.innerHTML = 'y'
	let left_bracket = document.createElement('mo')
	left_bracket.innerHTML = '('
	let y_inner_label = document.createElement('mi')
	y_inner_label.innerHTML = 'x'
	let right_bracket = document.createElement('mo')
	right_bracket.innerHTML = ')'
	let equate = document.createElement('mo')
	equate.innerHTML = '='
	save_formula.prepend(equate)
	save_formula.prepend(right_bracket)
	save_formula.prepend(y_inner_label)
	save_formula.prepend(left_bracket)
	save_formula.prepend(y_label)
	result.appendChild(save_formula)
	result.appendChild(document.createElement('br'))
	result.appendChild(document.createElement('br'))
	calculations.map(answer => {
		result.innerHTML += "Точность: "
		result.innerHTML += answer[0]
		result.appendChild(document.createElement('br'))
		result.innerHTML += "X = "
		result.innerHTML += answer[1].toFixed(6)
		result.innerHTML += " ± "
		result.innerHTML += answer[3].toFixed(6)
		result.appendChild(document.createElement('br'))
		result.innerHTML += "Количество итераций:"
		result.innerHTML += answer[4]
		result.appendChild(document.createElement('br'))
		let img = new Image();
		img.onload = function(){
		  // execute drawImage statements here
		};
		img.src = answer[5];
		result.appendChild(img);
		result.appendChild(document.createElement('br'))
	});

	output.prepend(result);
}

async function buttonClick(data, move) {
	let text = document.getElementById('equation').value;
	document.getElementById('equation').value =
		[text.slice(0, cursorPosition), data, text.slice(cursorPosition),].join('');
	document.getElementById('equation').focus();
	cursorPosition += move;
	document.getElementById('equation').selectionStart = cursorPosition;
    document.getElementById('equation').selectionEnd = cursorPosition;
	parse(document.getElementById('equation').value);
}

async function clearInput() {
	document.getElementById('equation').value = ''
	document.getElementById('equation_showcase').innerHTML = ''
}

async function toMathMl(expression) {
	let str = ``;
	if (expression.Binary) {
		if (expression.Binary.operator === '/') {
			str += `<mfrac>`
			if (expression.Binary.left.Expression) {
				str += `<mrow>`
				str += `<mo>(</mo>`
				await toMathMl(expression.Binary.left.Expression).then(e => str = str +  e)
				str += `<mo>)</mo>`
				str += `</mrow>`
			} else
				await toMathMl(expression.Binary.left).then(e => str = str +  e)
			if (expression.Binary.right.Expression) {
				str += `<mrow>`
				str += `<mo>(</mo>`
				await toMathMl(expression.Binary.right.Expression).then(e => str = str +  e)
				str += `<mo>)</mo>`
				str += `</mrow>`
			} else
				await toMathMl(expression.Binary.right).then(e => str = str +  e)
			str += `</mfrac>`
		} else {
			str += `<mrow>`
			if (expression.Binary.left.Expression) {
				str += `<mrow>`
				str += `<mo>(</mo>`
				await toMathMl(expression.Binary.left.Expression).then(e => str = str +  e)
				str += `<mo>)</mo>`
				str += `</mrow>`
			} else
				await toMathMl(expression.Binary.left).then(e => str = str +  e)
			str += `<mo>`
			str = str + expression.Binary.operator
			str += `</mo>`
			if (expression.Binary.right.Expression) {
				str += `<mrow>`
				str += `<mo>(</mo>`
				await toMathMl(expression.Binary.right.Expression).then(e => str = str +  e)
				str += `<mo>)</mo>`
				str += `</mrow>`
			} else
				await toMathMl(expression.Binary.right).then(e => str = str +  e)
			str += `</mrow>`
		}
	} else if (expression.Unary) {
		str += `<mo>`
		str = str + expression.Unary.operator
		str += `</mo>`
		if (expression.Unary.expression.Expression) {
			str += `<mrow>`
			str += `<mo>(</mo>`
			await toMathMl(expression.Unary.expression.Expression).then(e => str = str +  e)
			str += `<mo>)</mo>`
			str += `</mrow>`
		} else
			await toMathMl(expression.Unary.expression).then(e => str = str +  e)
	} else if (expression.FunctionCall) {
		if (expression.FunctionCall.name === `factorial`) {
			if (expression.FunctionCall.args[0].Expression) {
				str += `<mrow>`
				str += `<mo>(</mo>`
				await toMathMl(expression.FunctionCall.args[0].Expression).then(e => str = str + e)
				str += `<mo>)</mo>`
				str += `</mrow>`
			} else
				await toMathMl(expression.FunctionCall.args[0]).then(e => str = str +  e)
			str += `<mo>`
			str = str + '!'
			str += `</mo>`
		} else if (expression.FunctionCall.name === `sqrt`) {
			if (expression.FunctionCall.args.length === 1 ||
			   (expression.FunctionCall.args[expression.FunctionCall.args.length - 1].Number === '2')) {
				str += `<msqrt>`
				str += `<mrow>`
				if (expression.FunctionCall.args[0].Expression) {
					str += `<mrow>`
					str += `<mo>(</mo>`
					await toMathMl(expression.FunctionCall.args[0].Expression).then(e => str += e)
					str += `<mo>)</mo>`
					str += `</mrow>`
				} else
					await toMathMl(expression.FunctionCall.args[0]).then(e => str += e)
				str += `</mrow>`
				str += `</msqrt>`
			} else {
				str += `<mrow class="absolute"><msub class="down"><mi></mi>`
				if (expression.FunctionCall.args[expression.FunctionCall.args.length - 1].Expression) {
					str += `<mrow>`
					str += `<mo>(</mo>`
					await toMathMl(expression.FunctionCall.args[expression.FunctionCall.args.length - 1].Expression).then(e => str += e)
					str += `<mo>)</mo>`
					str += `</mrow>`
				} else
					await toMathMl(expression.FunctionCall.args[expression.FunctionCall.args.length - 1]).then(e => str += e)
				str += `</msub>`
				str += `<msqrt><mrow>`
				if (expression.FunctionCall.args[0].Expression) {
						str += `<mrow>`
						str += `<mo>(</mo>`
						await toMathMl(expression.FunctionCall.args[0].Expression).then(e => str += e)
						str += `<mo>)</mo>`
						str += `</mrow>`
					} else
						await toMathMl(expression.FunctionCall.args[0]).then(e => str += e)

				str += `</mrow>`
				str += `</msqrt></mrow>`
			}
		}  else if (expression.FunctionCall.name === `pow`) {
			str += `<msup>`
			if (expression.FunctionCall.args[0].Expression) {
				str += `<mrow>`
				str += `<mo>(</mo>`
				await toMathMl(expression.FunctionCall.args[0].Expression).then(e => str = str +  e)
				str += `<mo>)</mo>`
				str += `</mrow>`
			} else
				await toMathMl(expression.FunctionCall.args[0]).then(e => str = str +  e)
			if (expression.FunctionCall.args[1].Expression) {
				str += `<mrow>`
				str += `<mo>(</mo>`
				await toMathMl(expression.FunctionCall.args[1].Expression).then(e => str = str +  e)
				str += `<mo>)</mo>`
				str += `</mrow>`
			} else
				await toMathMl(expression.FunctionCall.args[1]).then(e => str = str +  e)
			str += `</msub>`
		} else if (expression.FunctionCall.name === `log`) {
			str += `<mrow><msub>`
			str += `<mi>${expression.FunctionCall.name}</mi>`
			if (expression.FunctionCall.args[1].Expression) {
				str += `<mrow>`
				str += `<mo>(</mo>`
				await toMathMl(expression.FunctionCall.args[1].Expression).then(e => str = str +  e)
				str += `<mo>)</mo>`
				str += `</mrow>`
			} else
				await toMathMl(expression.FunctionCall.args[1]).then(e => str = str +  e)
			str += `</msub>`
			if (expression.FunctionCall.args[0].Expression) {
				str += `<mrow>`
				str += `<mo>(</mo>`
				await toMathMl(expression.FunctionCall.args[0].Expression).then(e => str = str +  e)
				str += `<mo>)</mo>`
				str += `</mrow>`
			} else
				await toMathMl(expression.FunctionCall.args[0]).then(e => str = str +  e)
			str += `</mrow>`
		} else {
			str += `<mrow>`
			str += "<mo>" + expression.FunctionCall.name + "</mo>"
			for (let arg_index = 0; arg_index < expression.FunctionCall.args.length - 1; arg_index++) {
				if (expression.FunctionCall.args[arg_index].Expression) {
					str += `<mrow>`
					str += `<mo>(</mo>`
					await toMathMl(expression.FunctionCall.args[arg_index].Expression).then(e => str += e)
					str += `<mo>)</mo>`
					str += `</mrow>`
				} else
					await toMathMl(expression.FunctionCall.args[arg_index]).then(e => str += e)
				if (arg_index < expression.FunctionCall.args.length - 2) {
					str += `<mo>`
					str += ','
					str += `</mo>`
				}
			}
			if (expression.FunctionCall.args[expression.FunctionCall.args.length - 1].Expression) {
				str += `<mrow>`
				str += `<mo>(</mo>`
				await toMathMl(expression.FunctionCall.args[expression.FunctionCall.args.length - 1].Expression).then(e => str += e)
				str += `<mo>)</mo>`
				str += `</mrow>`
			} else
				await toMathMl(expression.FunctionCall.args[expression.FunctionCall.args.length - 1]).then(e => str += e)
			str += `</mrow>`
		}
	} else if (expression.Identifier) {
		str += `<mi>`
		str += expression.Identifier === `pi` ? 'π' : expression.Identifier
		str += `</mi>`
	} else if (expression.Number) {
		str += `<mn>`
		str += expression.Number
		str += `</mn>`
	}
	return str
}

const convertStringToMath = async htmlString => {
    const parser = new DOMParser();
    const html = parser.parseFromString(htmlString, 'text/html');
    return html.body.innerHTML;
}