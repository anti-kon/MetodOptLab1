async function getBruteForceMethodResult() {
    let consumersNum = parseInt(document.getElementById('consumers_num').value);
    let suppliersNum = parseInt(document.getElementById('suppliers_num').value);
    if ((isNaN(consumersNum) || consumersNum <= 0 || consumersNum > 20) ||
        (isNaN(suppliersNum) || suppliersNum <= 0 || suppliersNum > 20)){
        return;
    }

    let costMatrix = new Array(suppliersNum);
    let proposalVector = new Array(suppliersNum);
    let demandVector = new Array(consumersNum);

    for (let supplier_index = 0; supplier_index < costMatrix.length; supplier_index++)
        costMatrix[supplier_index] = new Array(consumersNum);

    for (let supplier_index = 0; supplier_index < costMatrix.length; supplier_index++)
        for (let consumer_index = 0; consumer_index < costMatrix[supplier_index].length; consumer_index++) {
            let celLValue = parseInt(document.
            getElementById(`row${supplier_index}Column${consumer_index}Input`).value);
            costMatrix[supplier_index][consumer_index] = isNaN(celLValue) ? 0 : celLValue;
        }

    for (let supplier_index = 0; supplier_index < suppliersNum; supplier_index++) {
        let celLValue = parseInt(document.
        getElementById(`stocks${supplier_index}Input`).value);
        proposalVector[supplier_index] = isNaN(celLValue) ? 0 : celLValue;
    }

    for (let consumer_index = 0; consumer_index < consumersNum; consumer_index++) {
        let celLValue = parseInt(document.
        getElementById(`demand${consumer_index}Input`).value);
        demandVector[consumer_index] = isNaN(celLValue) ? 0 : celLValue;
    }

    let output = document.getElementById("output");
    let result = document.createElement('div');
    result.className = 'result';
    let calculations_brute_force = await eel.get_brute_force_result(costMatrix, proposalVector, demandVector)();
    let calculations = await eel.get_potentials_method_result(costMatrix, proposalVector, demandVector)();
    console.log(calculations_brute_force);

    let answerMatrixValue = new Array(calculations[10].length - 1);
    for (let rowIndex = 0; rowIndex < answerMatrixValue.length; rowIndex++) {
        answerMatrixValue[rowIndex] = new Array(calculations[10][rowIndex].length - 1);
        for (let columnIndex = 0; columnIndex < answerMatrixValue[rowIndex].length; columnIndex++)
            answerMatrixValue[rowIndex][columnIndex] = calculations[10][rowIndex][columnIndex];
    }

    let answerValue = calculations[10][calculations[10].length - 1][calculations[10][calculations[10].length - 1].length - 1]

    result.appendChild(document.createTextNode("Ответ:"));
    let answerMatrix = await generateMatrixView(answerMatrixValue, "Xопт");
    answerMatrix.style.marginLeft = '20px';
    answerMatrix.style.marginTop = '10px';
    result.appendChild(answerMatrix);
    let answer = document.createElement('div');
    answer.style.display = 'flex';
    answer.style.width = 'max-content';
    answer.innerHTML = `Smin = ${answerValue}`;
    answer.style.marginLeft = '20px';
    result.appendChild(answer);

    let startLabel = document.createElement('p')
    startLabel.innerHTML = "Исходная задача:";
    result.appendChild(startLabel)
    let start_table = await generateTaskMatrix(calculations[0], calculations[1], calculations[2], NaN,
        false, [], "Запас", "Спрос", false, []);
    start_table.style.marginLeft = '20px';
    start_table.style.marginTop = '10px'
    result.appendChild(start_table);

    let solveLabel = document.createElement('p')

    let canonLabel = document.createElement('p')
    canonLabel.innerHTML = "Условие в каноническом виде задачи линейного программирования:";
    let canonMatrix = await generateMatrixView(calculations_brute_force[5], "A");
    canonMatrix.style.marginLeft = '20px';
    canonMatrix.style.marginTop = '10px';
    result.appendChild(canonLabel)
    result.appendChild(canonMatrix);
    canonMatrix.appendChild(document.createElement('br'));
    canonMatrix.innerHTML += ("B = [" + calculations_brute_force[6][0]);
    for (let i = 1; i < calculations_brute_force[6].length; i++)
        canonMatrix.innerHTML += (" " + calculations_brute_force[6][i]);
    canonMatrix.innerHTML += ("]");
    canonMatrix.appendChild(document.createElement('br'));
    canonMatrix.innerHTML += ("F(X) = [" + calculations_brute_force[7][0]);
    for (let i = 1; i < calculations_brute_force[7].length; i++)
        canonMatrix.innerHTML += (" " + calculations_brute_force[7][i]);
    canonMatrix.innerHTML += ("]");

    let iterationsLabel = document.createElement('p')
    iterationsLabel.innerHTML = `Количество итераций: ${calculations_brute_force[11]}`;
    result.appendChild(iterationsLabel);

    output.prepend(result);
}

document.getElementById('brute_force_calculate_btn').addEventListener('click', async () => {
    await getBruteForceMethodResult();
})


async function getPotentialsMethodResult() {
    let consumersNum = parseInt(document.getElementById('consumers_num').value);
    let suppliersNum = parseInt(document.getElementById('suppliers_num').value);
    if ((isNaN(consumersNum) || consumersNum <= 0 || consumersNum > 20) ||
        (isNaN(suppliersNum) || suppliersNum <= 0 || suppliersNum > 20)){
        return;
    }

    let costMatrix = new Array(suppliersNum);
    let proposalVector = new Array(suppliersNum);
    let demandVector = new Array(consumersNum);

    for (let supplier_index = 0; supplier_index < costMatrix.length; supplier_index++)
        costMatrix[supplier_index] = new Array(consumersNum);

    for (let supplier_index = 0; supplier_index < costMatrix.length; supplier_index++)
        for (let consumer_index = 0; consumer_index < costMatrix[supplier_index].length; consumer_index++) {
            let celLValue = parseInt(document.
                getElementById(`row${supplier_index}Column${consumer_index}Input`).value);
            costMatrix[supplier_index][consumer_index] = isNaN(celLValue) ? 0 : celLValue;
        }

    for (let supplier_index = 0; supplier_index < suppliersNum; supplier_index++) {
        let celLValue = parseInt(document.
            getElementById(`stocks${supplier_index}Input`).value);
        proposalVector[supplier_index] = isNaN(celLValue) ? 0 : celLValue;
    }

    for (let consumer_index = 0; consumer_index < consumersNum; consumer_index++) {
        let celLValue = parseInt(document.
            getElementById(`demand${consumer_index}Input`).value);
        demandVector[consumer_index] = isNaN(celLValue) ? 0 : celLValue;
    }

    let output = document.getElementById("output");
    let result = document.createElement('div');
    result.className = 'result';
    let calculations = await eel.get_potentials_method_result(costMatrix, proposalVector, demandVector)();
    console.log(calculations)

    let answerMatrixValue = new Array(calculations[10].length - 1);
    for (let rowIndex = 0; rowIndex < answerMatrixValue.length; rowIndex++) {
        answerMatrixValue[rowIndex] = new Array(calculations[10][rowIndex].length - 1);
        for (let columnIndex = 0; columnIndex < answerMatrixValue[rowIndex].length; columnIndex++)
            answerMatrixValue[rowIndex][columnIndex] = calculations[10][rowIndex][columnIndex];
    }

    let answerValue = calculations[10][calculations[10].length - 1][calculations[10][calculations[10].length - 1].length - 1]

    result.appendChild(document.createTextNode("Ответ:"));
    let answerMatrix = await generateMatrixView(answerMatrixValue, "Xопт");
    answerMatrix.style.marginLeft = '20px';
    answerMatrix.style.marginTop = '10px';
    result.appendChild(answerMatrix);
    let answer = document.createElement('div');
    answer.style.display = 'flex';
    answer.style.width = 'max-content';
    answer.innerHTML = `Smin = ${answerValue}`;
    answer.style.marginLeft = '20px';
    result.appendChild(answer);

    let startLabel = document.createElement('p')
    startLabel.innerHTML = "Исходная задача:";
    result.appendChild(startLabel)
    let start_table = await generateTaskMatrix(calculations[0], calculations[1], calculations[2], NaN,
        false, [], "Запас", "Спрос", false, []);
    start_table.style.marginLeft = '20px';
    start_table.style.marginTop = '10px'
    result.appendChild(start_table);

    let solveLabel = document.createElement('p')
    solveLabel.innerHTML = "Решение:";
    result.appendChild(solveLabel)

    if (calculations[3] === true)  {
        let str = calculations[1][0].toString();
        for (let consumersIndex = 1; consumersIndex < calculations[1].length; consumersIndex++)
            str += (" + " + calculations[1][consumersIndex]);
        str += (" = " + calculations[2][0]);
        for (let suppliersIndex = 1; suppliersIndex < calculations[2].length; suppliersIndex++)
            str += (" + " + calculations[2][suppliersIndex]);

        let openCloseCalculations = document.createElement('p');
        openCloseCalculations.innerHTML += "Суммарные запасы продукции у поставщиков равны суммарной потребности потребителей.";
        openCloseCalculations.appendChild(document.createElement('br'));
        openCloseCalculations.innerHTML += str;
        openCloseCalculations.appendChild(document.createElement('br'));
        openCloseCalculations.innerHTML += "Задача закрытого вида.";

        result.appendChild(openCloseCalculations);
    } else {
        let supplierSum = 0
        for (let supplier_index = 0; supplier_index < calculations[1].length; supplier_index++)
            supplierSum += calculations[1][supplier_index];
        let consumerSum = 0
        for (let consumer_index = 0; consumer_index < calculations[2].length; consumer_index++)
            consumerSum += calculations[2][consumer_index];

        let str = calculations[1][0].toString();
        for (let consumersIndex = 1; consumersIndex < calculations[1].length; consumersIndex++)
            str += (" + " + calculations[1][consumersIndex]);
        str += ((supplierSum > consumerSum ? " ≤ " : " ≥ ")  + calculations[2][0]);
        for (let suppliersIndex = 1; suppliersIndex < calculations[2].length; suppliersIndex++)
            str += (" + " + calculations[2][suppliersIndex]);

        let openCloseCalculations = document.createElement('p');
        if (supplierSum > consumerSum)
            openCloseCalculations.innerHTML += "Суммарные запасы продукции у поставщиков больше суммарной потребности потребителей.";
        else
            openCloseCalculations.innerHTML += "Суммарные запасы продукции у поставщиков меньше суммарной потребности потребителей.";
        openCloseCalculations.appendChild(document.createElement('br'));
        openCloseCalculations.innerHTML += str;
        openCloseCalculations.appendChild(document.createElement('br'));
        openCloseCalculations.innerHTML += "Задача открытого вида. Привёдем её к закрытому виду.";
        openCloseCalculations.appendChild(document.createElement('br'));
        openCloseCalculations.innerHTML += "";
        if (supplierSum > consumerSum)
            openCloseCalculations.innerHTML += "Введем в рассмотрение фиктивного потребителя. Получили следующее условие:";
        else
            openCloseCalculations.innerHTML += "Введем в рассмотрение фиктивного поставщика. Получили следующее условие:"

        result.appendChild(openCloseCalculations);

        let closeTask = await generateTaskMatrix(calculations[4], calculations[5], calculations[6],
            NaN, false, [], "Запас", "Спрос", false, []);
        closeTask.style.marginLeft = '20px';
        result.appendChild(closeTask);
    }

    let start_matrix = new Array(calculations[8].length);
    let start_proposal_vector = new Array(calculations[8].length);
    let start_demand_vector = new Array(calculations[8][0].length);
    for (let supplier_index = 0; supplier_index < calculations[8].length; supplier_index++) {
        start_matrix[supplier_index] = new Array(calculations[8][supplier_index].length);
        start_proposal_vector[supplier_index] = 0;
        for (let consumer_index = 0; consumer_index < calculations[8][supplier_index].length; consumer_index++)
            start_matrix[supplier_index][consumer_index] =
                calculations[8][supplier_index][consumer_index] === 0 ? NaN :
                    calculations[7][supplier_index][consumer_index];
    }

    for (let consumer_index = 0; consumer_index < calculations[8][0].length; consumer_index++)
        start_demand_vector[consumer_index] = 0;

    let startPlanLabel = document.createElement('p')
    startPlanLabel.innerHTML = "Построим начальный опорный план методом Северо-Западного угла:";
    result.appendChild(startPlanLabel)
    let startPlan = await generateTaskMatrix(start_matrix, start_proposal_vector, start_demand_vector,
        calculations[7][calculations[8].length][calculations[8][0].length], true, calculations[8],
        "Запас", "Спрос", false, []);
    startPlan.style.marginLeft = '20px';
    result.appendChild(startPlan);

    for (let step = 0; step < calculations[9].length; step++) {
        let step_matrix = new Array(calculations[9][step][1].length);
        let step_proposal_vector = calculations[9][step][2];
        let step_demand_vector = calculations[9][step][3];
        for (let supplier_index = 0; supplier_index < calculations[9][step][1].length; supplier_index++) {
            step_matrix[supplier_index] = new Array(calculations[9][step][1][supplier_index].length);
            for (let consumer_index = 0; consumer_index < calculations[9][step][1][supplier_index].length; consumer_index++)
                step_matrix[supplier_index][consumer_index] =
                    calculations[9][step][1][supplier_index][consumer_index] === 0 ? NaN :
                        calculations[9][step][0][supplier_index][consumer_index];
        }

        let planCheckLabel = document.createElement('p');
        planCheckLabel.innerHTML = "Проверим опорный план на оптимальность";
        planCheckLabel.appendChild(document.createElement('br'));
        planCheckLabel.innerHTML += `<span class="tab"></span>Каждому поставщику Ai ставим в соответствие некоторое число Ui, называемое потенциалом поставщика.`;
        planCheckLabel.appendChild(document.createElement('br'));
        planCheckLabel.innerHTML += `<span class="tab"></span>Каждому потребителю Bj ставим в соответствие некоторое число Vj , называемое потенциалом потребителя.`;
        result.appendChild(planCheckLabel);

        let potentialsLabel = document.createElement('p');
        potentialsLabel.innerHTML = "Последовательно найдем значения потенциалов. Составим СЛАУ вида: Vj - Ui = Cij, где Cij - тарифы, стоящие в заполненных клетках таблицы условий транспортной задачи.";
        potentialsLabel.appendChild(document.createElement('br'));
        for (let supplier_index = 0; supplier_index < calculations[9][step][1].length; supplier_index++) {
            for (let consumer_index = 0; consumer_index < calculations[9][step][1][supplier_index].length; consumer_index++)
                if (calculations[9][step][1][supplier_index][consumer_index] !== 0) {
                    potentialsLabel.innerHTML += (`<span class="tab"></span>V${(supplier_index + 1)} - U${(consumer_index + 1)} = ${calculations[4][supplier_index][consumer_index]}`);
                    potentialsLabel.appendChild(document.createElement('br'));
                }
        }
        potentialsLabel.innerHTML += "Значение одного потенциала необходимо задать. Пусть U1 = 0. Полученные значения потенциалов:";
        potentialsLabel.appendChild(document.createElement('br'));
        for (let supplier_potential_index = 0; supplier_potential_index < calculations[9][step][2].length; supplier_potential_index++)
            potentialsLabel.innerHTML += `<span class="tab"></span>U${(supplier_potential_index + 1)} = ${calculations[9][step][2][supplier_potential_index]}`

        for (let consumers_potential_index = 0; consumers_potential_index < calculations[9][step][3].length; consumers_potential_index++)
            potentialsLabel.innerHTML += `<span class="tab"></span>V${(consumers_potential_index + 1)} = ${calculations[9][step][3][consumers_potential_index]}`
        result.appendChild(potentialsLabel);

        let deltasLabel = document.createElement('p');
        deltasLabel.innerHTML = "Для незадействованных маршрутов определим числа Δij = Vj + Ui - Cij, где Cij - тарифы, стоящие в заполненных клетках таблицы условий транспортной задачи.";
        deltasLabel.appendChild(document.createElement('br'));
        for (let supplier_index = 0; supplier_index < calculations[9][step][2].length; supplier_index++) {
            for (let consumers_index = 0; consumers_index < calculations[9][step][3].length; consumers_index++) {
                if (calculations[9][step][4][supplier_index][consumers_index] !== 0) {
                    deltasLabel.innerHTML += `<span class="tab"></span>Δ${(supplier_index + 1)}${(consumers_index + 1)} = ${calculations[9][step][3][consumers_index]} + ${calculations[9][step][2][supplier_index]} - ${calculations[4][supplier_index][consumers_index]} = ${calculations[9][step][4][supplier_index][consumers_index]}`
                    deltasLabel.appendChild(document.createElement('br'));
                }
            }
        }

        result.appendChild(potentialsLabel);
        result.appendChild(deltasLabel);

        let isOptimal = true;
        for (let supplier_index = 0; supplier_index < calculations[9][step][2].length; supplier_index++)
            for (let consumers_index = 0; consumers_index < calculations[9][step][3].length; consumers_index++)
                if (calculations[9][step][4][supplier_index][consumers_index] !== 0 && calculations[9][step][4][supplier_index][consumers_index] > 0)
                    isOptimal = false;

        let afterStartCheck = document.createElement('p');
        afterStartCheck.innerHTML = "Матрица с потенциалами, оценками и задействованными путями.";
        result.appendChild(afterStartCheck);

        let potentialsMatrix = await generateTaskMatrix(step_matrix, step_proposal_vector, step_demand_vector,
            calculations[9][step][0][calculations[9][step][1].length][calculations[9][step][1][0].length],
            true, calculations[9][step][1], "U", "V", true,
            calculations[9][step][4]);
        potentialsMatrix.style.marginLeft = '20px';
        potentialsMatrix.style.marginBottom = '20px';
        result.appendChild(potentialsMatrix);

        if (isOptimal) {
            let optimalLabel = document.createElement('p');
            optimalLabel.innerHTML = "Положительных оценок нет. Следовательно решение оптимально.";
            result.appendChild(optimalLabel);
        } else {
            let optimalLabel = document.createElement('p');
            optimalLabel.innerHTML = "Есть полжительные оценки. Следовательно, возможно получить новое решение, как минимум, не хуже имеющегося.";
            result.appendChild(optimalLabel);

            result.appendChild(document.createTextNode(`ШАГ №: ${(step + 1)}`));
            result.appendChild(document.createElement('br'));
            let cycleLabel = document.createElement('p');
            cycleLabel.innerHTML = `Выберем ячейку A${calculations[9][step][5][0][0] + 1}B${calculations[9][step][5][0][1] + 1}, так как её оценка наибольшая.`;
            cycleLabel.appendChild(document.createElement('br'));
            cycleLabel.innerHTML += `Построим цикл пересчёта, используя только горизонтальные и вертикальные перемещения соединим ячейки с задействованными путями, так чтобы вернуться в исходную ячейку A${calculations[9][step][5][0][0] + 1}B${calculations[9][step][5][0][1] + 1}.`;
            cycleLabel.appendChild(document.createElement('br'));
            cycleLabel.innerHTML += "Элементы цикла пересчёта взятого с \"+\" обозначим — ";
            let redLabel = document.createElement('span');
            redLabel.innerHTML = "красным";
            redLabel.style.color = "rgb(204,154,154)";
            cycleLabel.appendChild(redLabel);
            cycleLabel.appendChild(document.createElement('br'));
            cycleLabel.innerHTML += "Элементы цикла пересчёта взятого с \"-\" обозначим — ";
            let blueLabel = document.createElement('span');
            blueLabel.innerHTML = "синим";
            blueLabel.style.color = "rgb(141,148,201)";
            cycleLabel.appendChild(blueLabel);
            result.appendChild(cycleLabel);

            for (let index = 0; index < calculations[9][step][6].length; index++){
                let cycleStepMatrix = result.appendChild(await generateTaskMatrix(step_matrix, step_proposal_vector, step_demand_vector,
                    calculations[9][step][0][calculations[9][step][1].length][calculations[9][step][1][0].length],
                    true, calculations[9][step][1], "U", "V", false,
                    [], true, calculations[9][step][6][index][0]));
                cycleStepMatrix.style.marginLeft = '20px';
                result.appendChild(cycleStepMatrix);
            }

            let cycleMatrix = result.appendChild(await generateTaskMatrix(step_matrix, step_proposal_vector, step_demand_vector,
                calculations[9][step][0][calculations[9][step][1].length][calculations[9][step][1][0].length],
                true, calculations[9][step][1], "U", "V", false,
                [], true, calculations[9][step][5]));
            cycleMatrix.style.marginLeft = '20px';
            result.appendChild(cycleMatrix);

            let minNegative = document.createElement('p');
            let min = calculations[9][step][0][calculations[9][step][5][1][0]][calculations[9][step][5][1][1]];
            minNegative.innerHTML = "Выберем клетку, взятую со знаком \"-\", содержащую минимальный коэффициент";
            minNegative.appendChild(document.createElement('br'));
            minNegative.innerHTML += "min(" + calculations[9][step][0][calculations[9][step][5][1][0]][calculations[9][step][5][1][1]];
            for (let cycleStep = 2; cycleStep < calculations[9][step][5].length; cycleStep++) {
                if (cycleStep % 2 !== 0) {
                    minNegative.innerHTML += ", " + calculations[9][step][0][calculations[9][step][5][cycleStep][0]][calculations[9][step][5][cycleStep][1]];
                    if (min > calculations[9][step][0][calculations[9][step][5][cycleStep][0]][calculations[9][step][5][cycleStep][1]])
                        min = calculations[9][step][0][calculations[9][step][5][cycleStep][0]][calculations[9][step][5][cycleStep][1]];
                }
            }
            minNegative.innerHTML += ") = ";
            minNegative.innerHTML += min.toString();
            result.appendChild(minNegative);


            let newPlanLabel = document.createElement('p');
            newPlanLabel.innerHTML = "Определим новый опорный план транспортной задачи:";
            cycleLabel.appendChild(newPlanLabel);
            let step_new_matrix = new Array(calculations[9][step + 1][1].length);
            let step_new_proposal_vector = new Array(calculations[9][step + 1][1].length);
            let step_new_demand_vector = new Array(calculations[9][step + 1][1][0].length);
            for (let supplier_index = 0; supplier_index < calculations[9][step + 1][1].length; supplier_index++) {
                step_new_matrix[supplier_index] = new Array(calculations[9][step + 1][1][supplier_index].length);
                step_new_proposal_vector[supplier_index] = 0;
                for (let consumer_index = 0; consumer_index < calculations[9][step + 1][1][supplier_index].length; consumer_index++) {
                    step_new_matrix[supplier_index][consumer_index] =
                        calculations[9][step + 1][1][supplier_index][consumer_index] === 0 ? NaN :
                            calculations[9][step + 1][0][supplier_index][consumer_index];
                    step_new_demand_vector[consumer_index] = 0;
                }
            }
            let newMatrix = result.appendChild(await generateTaskMatrix(step_new_matrix, step_new_proposal_vector, step_new_demand_vector,
                calculations[9][step + 1][0][calculations[9][step + 1][1].length][calculations[9][step + 1][1][0].length],
                true, calculations[9][step + 1][1], "Запас", "Спрос", false,
                [], false, calculations[9][step][5]));
            newMatrix.style.marginLeft = '20px';
            result.appendChild(newMatrix);
        }
        output.prepend(result);
    }
}

async function generateMatrixView(matrix, matrixLabel) {
    let matrixBody = document.createElement('div');
    matrixBody.className = 'equation';
    matrixBody.innerHTML = `${matrixLabel} =`;
    let matrixView = document.createElement('table');
    matrixView.className = 'matrix';
    matrixView.style.marginLeft = '5px';
    matrixBody.appendChild(matrixView);
    let matrixContent = document.createElement('tbody');

    for (let rowIndex = 0; rowIndex < matrix.length; rowIndex++) {
        let matrixRow = document.createElement('tr');
        for (let columnIndex = 0; columnIndex < matrix[rowIndex].length; columnIndex++) {
            let numCell = document.createElement('td');
            numCell.innerHTML = matrix[rowIndex][columnIndex];
            matrixRow.appendChild(numCell);
        }
        matrixContent.appendChild(matrixRow);
    }
    matrixView.appendChild(matrixContent);

    // matrix_body.style.display = "flex";
    // matrix_body.style.width = "max_content";
    // let matrix_label = document.createElement('div');
    // matrix_label.innerHTML = "X = ";
    // matrix_label.style.display = 'flex';
    // matrix_label.style.height = 'auto';
    // matrix_label.style.maxHeight = '100%';
    // matrix_label.style.marginRight = '5px';
    // matrix_label.style.justifyContent = 'center';
    // matrix_label.style.alignItems = 'center';
    // matrix_body.appendChild(matrix_label);
    // let matrix_view = document.createElement('div');
    // matrix_view.style.display = "grid";
    // matrix_view.style.gridTemplateColumns = `repeat(${matrix[0].length + 2}, max-content)`;

    // matrix_body.appendChild(matrix_view);
    return matrixBody;
}

async function handleFileLoad(event) {
    const file = document.getElementById('file_input').files[0];
    const reader = new FileReader();

    reader.addEventListener(
        "load",
        () => {
            let data = reader.result.match(/(-?\d+(\.\d+)?)/g).map(v => +v);
            let consumersNum = data[0];
            let suppliersNum = data[1];

            document.getElementById('consumers_num').value = consumersNum;
            document.getElementById('suppliers_num').value = suppliersNum;

            addRelationshipsMatrix();

            for (let suppliersIndex = 0; suppliersIndex < suppliersNum; suppliersIndex++)
                for (let consumersIndex = 0; consumersIndex < consumersNum; consumersIndex++)
                    document.getElementById(`row${suppliersIndex}Column${consumersIndex}Input`).value =
                        data[2 + suppliersIndex * consumersNum + consumersIndex + suppliersIndex];

            for (let consumersIndex = 0; consumersIndex < consumersNum; consumersIndex++)
                document.getElementById(`demand${consumersIndex}Input`).value =
                    data[data.length - consumersNum + consumersIndex];

            for (let suppliersIndex = 0; suppliersIndex < suppliersNum; suppliersIndex++)
                document.getElementById(`stocks${suppliersIndex}Input`).value =
                    data[2 + (suppliersIndex + 1) * consumersNum + suppliersIndex];
        },
        false,
    );

    if (file) {
        reader.readAsText(file);
    }
}

async function generateTaskMatrix(costMatrix, proposalVector, demandVector, sumValue, useMaskMatrix, maskMatrix,
                                  firstLabel, secondLabel, useDeltas, deltasMatrix, useCycle, cycleVector) {
    let table = document.createElement('div');
    table.className='table';

    let suppliersLabel = document.createElement('div');
    suppliersLabel.innerHTML="Поставщики";
    suppliersLabel.className = 'tableCell';
    table.appendChild(suppliersLabel);

    let consumersLegend = document.createElement('div');

    let consumersLabel = document.createElement('div');
    consumersLabel.innerHTML = "Потребители";
    consumersLabel.className = 'tableCell';
    consumersLegend.appendChild(consumersLabel);

    let consumersNamesRow = document.createElement('div');
    consumersNamesRow.className = 'tableRow';
    for (let consumer_index = 0; consumer_index < costMatrix[0].length; consumer_index++) {
        let consumerName = document.createElement('div');
        consumerName.innerHTML = `B${(consumer_index+1)}`;
        consumerName.className = 'tableCell';
        consumersNamesRow.appendChild(consumerName);
    }
    consumersLegend.appendChild(consumersNamesRow);

    table.appendChild(consumersLegend);

    let stocksLabel = document.createElement('div');
    stocksLabel.innerHTML=firstLabel;
    stocksLabel.className = 'tableCell';
    table.appendChild(stocksLabel);

    for (let supplier_index = 0; supplier_index < costMatrix.length; supplier_index++) {
        let supplierName = document.createElement('div');
        supplierName.innerHTML = `A${(supplier_index+1)}`;
        supplierName.className = 'tableCell';
        table.appendChild(supplierName);

        let tableRow = document.createElement('div');
        tableRow.className = 'tableRow';

        for (let consumer_index = 0; consumer_index < costMatrix[supplier_index].length; consumer_index++) {
            let tableCell = document.createElement('div');
            tableCell.className = 'tableCell';
            if (!isNaN(costMatrix[supplier_index][consumer_index]))
                tableCell.innerHTML = costMatrix[supplier_index][consumer_index];
            if (useMaskMatrix && maskMatrix[supplier_index][consumer_index] === 1)
                tableCell.style.background="#c5baaa";
            if (useDeltas && maskMatrix[supplier_index][consumer_index] === 0)
                tableCell.innerHTML = deltasMatrix[supplier_index][consumer_index];
            if (useCycle) {
                console.log(cycleVector)
                for (let cycleCellIndex = 0; cycleCellIndex < cycleVector.length; cycleCellIndex++) {
                    if (cycleVector[cycleCellIndex][0] === supplier_index &&
                        cycleVector[cycleCellIndex][1] === consumer_index) {
                        if (cycleCellIndex % 2 === 0)
                            tableCell.style.background = "rgb(234,202,202)";
                        else
                            tableCell.style.background = "rgb(202,206,234)";
                    }
                }
            }
            tableRow.appendChild(tableCell);
        }
        table.appendChild(tableRow);

        let stocksCell = document.createElement('div');
        stocksCell.className = 'tableCell';
        stocksCell.innerHTML = proposalVector[supplier_index];
        table.appendChild(stocksCell);
    }

    let demandLabel = document.createElement('div');
    demandLabel.innerHTML = secondLabel;
    demandLabel.className = 'tableCell';

    table.appendChild(demandLabel);

    let demandRow = document.createElement('div');
    demandRow.className = 'tableRow';

    for (let consumer_index = 0; consumer_index < demandVector.length; consumer_index++) {
        let demandCell = document.createElement('div');
        demandCell.className = 'tableCell';
        demandCell.innerHTML = demandVector[consumer_index];
        demandRow.appendChild(demandCell);
    }
    table.appendChild(demandRow);

    let sumCell = document.createElement('div');
    sumCell.className = 'tableCell';
    if (!isNaN(sumValue))
        sumCell.innerHTML = sumValue
    table.appendChild(sumCell);

    return table;
}

document.getElementById('potentials_calculate_btn').addEventListener('click', async () => {
    await getPotentialsMethodResult();
})

async function addRelationshipsMatrix() {
    let consumersNum = parseInt(document.getElementById('consumers_num').value);
    let suppliersNum = parseInt(document.getElementById('suppliers_num').value);
    if ((isNaN(consumersNum) || consumersNum <= 0 || consumersNum > 20) ||
        (isNaN(suppliersNum) || suppliersNum <= 0 || suppliersNum > 20)){
        return;
    }

    let container = document.getElementById('costMatrix');

    while (container.hasChildNodes())
        container.removeChild(container.lastChild);

    let suppliersLabel = document.createElement('div');
    suppliersLabel.innerHTML = "Поставщики";
    suppliersLabel.id = 'suppliersLabel';
    suppliersLabel.className = 'costMatrixText';
    container.appendChild(suppliersLabel);

    let consumersLegend = document.createElement('div');
    consumersLegend.id = 'consumersLegend';

    let consumersLabel = document.createElement('div');
    consumersLabel.innerHTML = "Потребители";
    consumersLabel.className = 'costMatrixText';
    consumersLegend.appendChild(consumersLabel);

    let consumersNamesRow = document.createElement('div');
    consumersNamesRow.className = 'costMatrixRow';
    for (let consumer_index = 0; consumer_index < consumersNum; consumer_index++) {
        let consumerName = document.createElement('div');
        consumerName.innerHTML = `B${(consumer_index+1)}`;
        consumerName.className = 'costMatrixText';
        consumerName.style.marginRight = 'auto';
        consumerName.style.marginLeft = 'auto';
        consumersNamesRow.appendChild(consumerName);
    }
    consumersLegend.appendChild(consumersNamesRow);

    container.appendChild(consumersLegend)

    let stocksLabel = document.createElement('div');
    stocksLabel.innerHTML = "Запас";
    stocksLabel.id = 'stocksLabel';
    stocksLabel.className = 'costMatrixText';
    container.appendChild(stocksLabel);

    for (let supplier_index = 0; supplier_index < suppliersNum; supplier_index++) {
        let supplierNameRow = document.createElement('div');
        supplierNameRow.className = 'costMatrixRow';
        supplierNameRow.id = `supplierNameRow${(supplier_index+1)}`

        let supplierName = document.createElement('div');
        supplierName.innerHTML = `A${(supplier_index+1)}`;
        supplierName.className = 'costMatrixText';
        supplierName.style.marginRight = 'auto';
        supplierName.style.marginLeft = 'auto';
        supplierNameRow.appendChild(supplierName);
        container.appendChild(supplierNameRow);

        let costMatrixRow = document.createElement('div');
        costMatrixRow.className = 'costMatrixRow';
        costMatrixRow.id = `costMatrixRow${supplier_index}`
        for (let consumer_index = 0; consumer_index < consumersNum; consumer_index++) {
            let costMatrixCell = document.createElement('div');
            costMatrixCell.className = 'costMatrixCell';

            let costMatrixCellInput = document.createElement('input');
            costMatrixCellInput.type = 'number';
            costMatrixCellInput.step = 'any';
            costMatrixCellInput.id = `row${supplier_index}Column${consumer_index}Input`;
            costMatrixCellInput.placeholder='0';
            costMatrixCell.appendChild(costMatrixCellInput);

            costMatrixRow.appendChild(costMatrixCell);
        }
        container.appendChild(costMatrixRow);

        let stocksCellRow = document.createElement('div');
        stocksCellRow.id = `stocksCellRow${supplier_index}`;
        stocksCellRow.className = 'costMatrixRow';

        let stocksCell = document.createElement('div');
        stocksCell.className = 'costMatrixCell';

        let stocksCellInput = document.createElement('input');
        stocksCellInput.type = 'number';
        stocksCellInput.step = 'any';
        stocksCellInput.id = `stocks${supplier_index}Input`;
        stocksCellInput.placeholder='0';

        stocksCell.appendChild(stocksCellInput);
        stocksCellRow.appendChild(stocksCell);
        container.appendChild(stocksCellRow);
    }

    let demandLabelRow = document.createElement('div');
    demandLabelRow.className = 'costMatrixRow';
    demandLabelRow.id = 'demandLabelRow';

    let demandLabel = document.createElement('div');
    demandLabel.innerHTML = "Спрос";
    demandLabel.className = 'costMatrixText';
    demandLabel.id = 'demandLabel';

    demandLabelRow.appendChild(demandLabel)
    container.appendChild(demandLabelRow);

    let demandRow = document.createElement('div');
    demandRow.id = `demandRow`;
    demandRow.className = 'costMatrixRow';

    for (let consumer_index = 0; consumer_index < consumersNum; consumer_index++) {
        let demandCell = document.createElement('div');
        demandCell.className = 'costMatrixCell';

        let demandCellInput = document.createElement('input');
        demandCellInput.type = 'number';
        demandCellInput.step = 'any';
        demandCellInput.id = `demand${consumer_index}Input`;
        demandCellInput.placeholder='0';
        demandCell.appendChild(demandCellInput);

        demandRow.appendChild(demandCell);
    }
    container.appendChild(demandRow);

    makeCoolView();
}

async function makeCoolView() {
    document.getElementById('demandRow').style.marginTop = '20px';
    document.getElementById('demandLabelRow').style.marginTop = '20px';
    document.getElementById('demandRow').style.marginBottom = '0px';
    document.getElementById('demandLabelRow').style.marginBottom = '0px';

    document.getElementById('costMatrixRow0').style.marginTop  = '20px';
    document.getElementById('supplierNameRow1').style.marginTop  = '20px';
    document.getElementById('stocksCellRow0').style.marginTop  = '20px';

    let container = document.getElementById('costMatrix');
    let appearance_block = document.createElement('div');
    appearance_block.style.position = 'absolute';
    appearance_block.style.width = '100%';
    appearance_block.style.height = '100%';
    appearance_block.style.zIndex = '-1';

    let suppliers_block = document.createElement('div');
    let suppliers_block_width = `${document.getElementById('suppliersLabel').offsetWidth + 10}px`;
    let suppliers_block_height = `${document.getElementById('consumersLegend').offsetHeight +
                                   (document.getElementById('supplierNameRow1').offsetHeight +
                                   (parseInt(getComputedStyle(document.getElementById('supplierNameRow1')).margin.match(/(\d+)/)[0]) - 15) * 2) *
                                    parseInt(document.getElementById('suppliers_num').value) + 15}px`;
    suppliers_block.className = 'blockDesign';
    suppliers_block.style.position = 'absolute';
    suppliers_block.style.width = suppliers_block_width;
    suppliers_block.style.height = suppliers_block_height;
    appearance_block.appendChild(suppliers_block);

    let consumers_block = document.createElement('div');
    let consumers_block_width = `${document.getElementById('consumersLegend').offsetWidth}px`;
    let consumers_block_height = `${document.getElementById('consumersLegend').offsetHeight}px`;
    consumers_block.className = 'blockDesign';
    consumers_block.style.position = 'absolute';
    consumers_block.style.width = consumers_block_width;
    consumers_block.style.height = consumers_block_height;
    consumers_block.style.left = `${document.getElementById('suppliersLabel').clientWidth + 15 + 10}px`;
    appearance_block.appendChild(consumers_block);

    let stocks_block = document.createElement('div');
    let stocks_block_width = `${document.getElementById('stocksLabel').offsetWidth + 10}px`;
    let stocks_block_height = `${document.getElementById('consumersLegend').offsetHeight +
    (document.getElementById('supplierNameRow1').offsetHeight +
        (parseInt(getComputedStyle(document.getElementById('supplierNameRow1')).margin.match(/(\d+)/)[0]) - 15) * 2) *
    parseInt(document.getElementById('suppliers_num').value) + 15}px`;
    stocks_block.className = 'blockDesign';
    stocks_block.style.position = 'absolute';
    stocks_block.style.width = stocks_block_width;
    stocks_block.style.height = stocks_block_height;
    stocks_block.style.right = '0px';
    appearance_block.appendChild(stocks_block);

    let demand_block = document.createElement('div');
    let demand_block_width = `${document.getElementById('demandLabel').offsetWidth + 
                                document.getElementById('demandRow').offsetWidth +
    parseInt(getComputedStyle(document.getElementById('demandRow')).margin.match(/(\d+)/)[0]) * 2 + 40}px`;
    let demand_block_height = `${document.getElementById('demandRow').offsetHeight + 5 * 2}px`;
    demand_block.className = 'blockDesign';
    demand_block.style.position = 'absolute';
    demand_block.style.width = demand_block_width;
    demand_block.style.height = demand_block_height;
    demand_block.style.bottom = '-5px';
    appearance_block.appendChild(demand_block);

    container.appendChild(appearance_block);
}