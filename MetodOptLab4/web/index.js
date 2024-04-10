let cursorPosition = 0;

async function setCursorPosition(value) {
	cursorPosition = value
}

async function getGradientMethodResult() {
	document.getElementById("loader").style.display='flex';
	let save_formula = document.getElementById('equation_showcase').cloneNode(true);
	save_formula.id = '';
	save_formula.className = '';
    let equation = document.getElementById('equation').value;
	let x = parseInt(document.getElementById('x_init').value);
	let y = parseInt(document.getElementById('y_init').value);
	let step = parseInt(document.getElementById('step_value').value);
	let x_min = parseInt(document.getElementById('x_min').value);
	let y_min = parseInt(document.getElementById('y_min').value);
	let x_max = parseInt(document.getElementById('x_max').value);
	let y_max = parseInt(document.getElementById('y_max').value);
	let x_split = parseInt(document.getElementById('x_split').value);
	let y_split = parseInt(document.getElementById('y_split').value);
	if (equation === '' || isNaN(x) || isNaN(y) || isNaN(x_min) || isNaN(y_min) ||
		isNaN(x_max) || isNaN(y_max) || isNaN(x_split) || isNaN(y_split)){
		document.getElementById("loader").style.display='none';
        return;
    }
	step = isNaN(step) ? 0.01 : step;

    let output = document.getElementById("output");
    let result = document.createElement('div');
    result.className = 'result';
	let calculations;
	try {
		calculations = await eel.get_gradient_method_result(equation, x, y, x_min, x_max, y_min, y_max, x_split, y_split, step)();
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
	let z_label = document.createElement('mi')
	z_label.innerHTML = 'z'
	let left_bracket = document.createElement('mo')
	left_bracket.innerHTML = '('
	let x_inner_label = document.createElement('mi')
	x_inner_label.innerHTML = 'x'
	let comma_inner_label = document.createElement('mo')
	comma_inner_label.innerHTML = ','
	let y_inner_label = document.createElement('mi')
	y_inner_label.innerHTML = 'y'
	let right_bracket = document.createElement('mo')
	right_bracket.innerHTML = ')'
	let equate = document.createElement('mo')
	equate.innerHTML = '='
	save_formula.prepend(equate)
	save_formula.prepend(right_bracket)
	save_formula.prepend(x_inner_label)
	save_formula.prepend(comma_inner_label)
	save_formula.prepend(y_inner_label)
	save_formula.prepend(left_bracket)
	save_formula.prepend(z_label)
	result.appendChild(save_formula)
	result.appendChild(document.createElement('br'))
	result.innerHTML += `Шаг: ${step}`
	result.appendChild(document.createElement('br'))
	console.log(calculations)
	calculations.map(answer => {
		result.innerHTML += "Точность: "
		result.innerHTML += answer[0]
		result.appendChild(document.createElement('br'))
		result.innerHTML += "Ответ: ("
		result.innerHTML += answer[1].toPrecision(1)
		result.innerHTML += ", "
		result.innerHTML += answer[2].toPrecision(1)
		result.innerHTML += ")"
		result.appendChild(document.createElement('br'))
		result.innerHTML += "Значение: "
		result.innerHTML += answer[3].toPrecision(1)
		result.appendChild(document.createElement('br'))
		let img1 = new Image();
		img1.onload = function(){
		  // execute drawImage statements here
		};
		img1.src = answer[4];
		result.appendChild(img1);
		result.appendChild(document.createElement('br'))
		let img2 = new Image();
		img2.onload = function(){
		  // execute drawImage statements here
		};
		img2.src = answer[5];
		result.appendChild(img2);
		result.appendChild(document.createElement('br'))
		let img3 = new Image();
		img3.onload = function(){
		  // execute drawImage statements here
		};
		img3.src = answer[6];
		result.appendChild(img3);
		result.appendChild(document.createElement('br'))
	});
	output.prepend(result);
}

async function getNewtonMethodResult() {
	document.getElementById("loader").style.display='flex';
	let save_formula = document.getElementById('equation_showcase').cloneNode(true);
	save_formula.id = '';
	save_formula.className = '';
    let equation = document.getElementById('equation').value;
	let x = parseInt(document.getElementById('x_init').value);
	let y = parseInt(document.getElementById('y_init').value);
	let delta = parseInt(document.getElementById('delta_value').value);
	let x_min = parseInt(document.getElementById('x_min').value);
	let y_min = parseInt(document.getElementById('y_min').value);
	let x_max = parseInt(document.getElementById('x_max').value);
	let y_max = parseInt(document.getElementById('y_max').value);
	let x_split = parseInt(document.getElementById('x_split').value);
	let y_split = parseInt(document.getElementById('y_split').value);
	if (equation === '' || isNaN(x) || isNaN(y) || isNaN(x_min) || isNaN(y_min) ||
		isNaN(x_max) || isNaN(y_max) || isNaN(x_split) || isNaN(y_split)){
		document.getElementById("loader").style.display='none';
        return;
    }
	delta = isNaN(delta) ? 0.01 : delta;

    let output = document.getElementById("output");
    let result = document.createElement('div');
    result.className = 'result';
	let calculations;
	try {
		calculations = await eel.get_newton_method_result(equation, x, y, x_min, x_max, y_min, y_max, x_split, y_split, delta)();
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
	let z_label = document.createElement('mi')
	z_label.innerHTML = 'z'
	let left_bracket = document.createElement('mo')
	left_bracket.innerHTML = '('
	let x_inner_label = document.createElement('mi')
	x_inner_label.innerHTML = 'x'
	let comma_inner_label = document.createElement('mo')
	comma_inner_label.innerHTML = ','
	let y_inner_label = document.createElement('mi')
	y_inner_label.innerHTML = 'y'
	let right_bracket = document.createElement('mo')
	right_bracket.innerHTML = ')'
	let equate = document.createElement('mo')
	equate.innerHTML = '='
	save_formula.prepend(equate)
	save_formula.prepend(right_bracket)
	save_formula.prepend(x_inner_label)
	save_formula.prepend(comma_inner_label)
	save_formula.prepend(y_inner_label)
	save_formula.prepend(left_bracket)
	save_formula.prepend(z_label)
	result.appendChild(save_formula)
	result.appendChild(document.createElement('br'))
	result.innerHTML += `Коэффициент сжатия шага: ${delta}`
	result.appendChild(document.createElement('br'))
	console.log(calculations)
	calculations.map(answer => {
		result.innerHTML += "Точность: "
		result.innerHTML += answer[0]
		result.appendChild(document.createElement('br'))
		result.innerHTML += "Ответ: ("
		result.innerHTML += answer[1].toPrecision(1)
		result.innerHTML += ", "
		result.innerHTML += answer[2].toPrecision(1)
		result.innerHTML += ")"
		result.appendChild(document.createElement('br'))
		result.innerHTML += "Значение: "
		result.innerHTML += answer[3].toPrecision(1)
		result.appendChild(document.createElement('br'))
		let img1 = new Image();
		img1.onload = function(){
		  // execute drawImage statements here
		};
		img1.src = answer[4];
		result.appendChild(img1);
		result.appendChild(document.createElement('br'))
		let img2 = new Image();
		img2.onload = function(){
		  // execute drawImage statements here
		};
		img2.src = answer[5];
		result.appendChild(img2);
		result.appendChild(document.createElement('br'))
		let img3 = new Image();
		img3.onload = function(){
		  // execute drawImage statements here
		};
		img3.src = answer[6];
		result.appendChild(img3);
		result.appendChild(document.createElement('br'))
	});
	output.prepend(result);
}

async function getBFGSResult() {
	document.getElementById("loader").style.display='flex';
	let save_formula = document.getElementById('equation_showcase').cloneNode(true);
	save_formula.id = '';
	save_formula.className = '';
    let equation = document.getElementById('equation').value;
	let x = parseInt(document.getElementById('x_init').value);
	let y = parseInt(document.getElementById('y_init').value);
	let x_min = parseInt(document.getElementById('x_min').value);
	let y_min = parseInt(document.getElementById('y_min').value);
	let x_max = parseInt(document.getElementById('x_max').value);
	let y_max = parseInt(document.getElementById('y_max').value);
	let x_split = parseInt(document.getElementById('x_split').value);
	let y_split = parseInt(document.getElementById('y_split').value);
	if (equation === '' || isNaN(x) || isNaN(y) || isNaN(x_min) || isNaN(y_min) ||
		isNaN(x_max) || isNaN(y_max) || isNaN(x_split) || isNaN(y_split)){
		document.getElementById("loader").style.display='none';
        return;
    }

    let output = document.getElementById("output");
    let result = document.createElement('div');
    result.className = 'result';
	let calculations;
	try {
		calculations = await eel.get_broyden_fletcher_goldfarb_shanno_method_result(equation, x, y, x_min, x_max, y_min, y_max, x_split, y_split)();
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
	let z_label = document.createElement('mi')
	z_label.innerHTML = 'z'
	let left_bracket = document.createElement('mo')
	left_bracket.innerHTML = '('
	let x_inner_label = document.createElement('mi')
	x_inner_label.innerHTML = 'x'
	let comma_inner_label = document.createElement('mo')
	comma_inner_label.innerHTML = ','
	let y_inner_label = document.createElement('mi')
	y_inner_label.innerHTML = 'y'
	let right_bracket = document.createElement('mo')
	right_bracket.innerHTML = ')'
	let equate = document.createElement('mo')
	equate.innerHTML = '='
	save_formula.prepend(equate)
	save_formula.prepend(right_bracket)
	save_formula.prepend(x_inner_label)
	save_formula.prepend(comma_inner_label)
	save_formula.prepend(y_inner_label)
	save_formula.prepend(left_bracket)
	save_formula.prepend(z_label)
	result.appendChild(save_formula)
	result.appendChild(document.createElement('br'))
	console.log(calculations)
	calculations.map(answer => {
		result.innerHTML += "Точность: "
		result.innerHTML += answer[0]
		result.appendChild(document.createElement('br'))
		result.innerHTML += "Ответ: ("
		result.innerHTML += answer[1].toPrecision(1)
		result.innerHTML += ", "
		result.innerHTML += answer[2].toPrecision(1)
		result.innerHTML += ")"
		result.appendChild(document.createElement('br'))
		result.innerHTML += "Значение: "
		result.innerHTML += answer[3].toPrecision(1)
		result.appendChild(document.createElement('br'))
		let img1 = new Image();
		img1.onload = function(){
		  // execute drawImage statements here
		};
		img1.src = answer[4];
		result.appendChild(img1);
		result.appendChild(document.createElement('br'))
		let img2 = new Image();
		img2.onload = function(){
		  // execute drawImage statements here
		};
		img2.src = answer[5];
		result.appendChild(img2);
		result.appendChild(document.createElement('br'))
		let img3 = new Image();
		img3.onload = function(){
		  // execute drawImage statements here
		};
		img3.src = answer[6];
		result.appendChild(img3);
		result.appendChild(document.createElement('br'))
	});	output.prepend(result);
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
			str += `<mrow><msup>`
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
			str += `</msub></mrow>`
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