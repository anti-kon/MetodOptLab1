async function buttonClick(data) {
	document.getElementById('equation').value += data
}

async function clearInput() {
	document.getElementById('equation').value = ''
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

		} else if (expression.FunctionCall.name === `sqrt`) {
			str += `<msqrt>`
			str += `<mrow>`
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
			str += `</msqrt>`
		}  else if (expression.FunctionCall.name === `pow`) {

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
	console.log(str)
	return str
}

const convertStringToMath = htmlString => {
    const parser = new DOMParser();
    const html = parser.parseFromString(htmlString, 'text/html');
    return html.body.innerHTML;
}