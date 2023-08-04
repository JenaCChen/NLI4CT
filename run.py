from train import Trainer, evidence_trainer
from flask import Flask, request, render_template, session, redirect, url_for
from load_data import parse_input, check_evidence

app = Flask(__name__)
app.secret_key = 'secret_key'
model_name = 'bert'
train_set = 'train.json'
dev_set = 'dev.json'
test_set = 'test.json'
epoch = 10
batch_size = 8
lr = 1e-5
mode = 2
ner = False

entailTrainer = Trainer(
    model_name=model_name,
    train_set=train_set,
    dev_set=dev_set,
    test_set=test_set,
    epoch=epoch,
    batch_size=batch_size,
    lr=lr,
    mode=mode,
    ner=ner,
    model_path='task1.pt',
)
entailTrainer.train()

evidenceTrainer = evidence_trainer(
    model_name=model_name,
    train_set=train_set,
    dev_set=dev_set,
    test_set=test_set,
    epoch=epoch,
    batch_size=batch_size,
    lr=lr,
    mode=mode,
    ner=ner,
    entail_trainer=entailTrainer,
    model_path='task2.pt'
)
evidenceTrainer.train()


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/task1", methods=["POST"])
def task1():
    text_input = request.form['text_input']
    if len(text_input.strip()) == 0:
        error_message = 'Please do not leave the statement blank.'
        return render_template('error.html', error_message=error_message)
    entail_label = entailTrainer.inference(text_input)
    session['statement'] = text_input
    session['entail_label'] = entail_label
    return render_template('task1.html', statement=text_input, entail_label=entail_label)


@app.route("/go_to_index")
def go_to_index():
    return redirect(url_for('index'))


@app.route("/task2", methods=["GET", "POST"])
def task2():
    statement = session['statement']
    entail_label = session['entail_label']
    section = request.form['section']
    evidence1 = request.form['evidence1']
    evidence2 = request.form['evidence2']
    session['evidence1'] = evidence1.strip()
    session['evidence2'] = evidence2.strip()

    if check_evidence(evidence1) is False:
        error_message = 'Invalid Primary Evidence ID!'
        return render_template('error.html', error_message=error_message)

    if len(evidence2) != 0:
        if check_evidence(evidence2) is False:
            error_message = 'Invalid Secondary Evidence ID!'
            return render_template('error.html', error_message=error_message)

    evidence1_list, evidence2_list, processed_evidence1_list, processed_evidence2_list = parse_input(statement,
                                                                                                     entail_label,
                                                                                                     section, evidence1,
                                                                                                     evidence2)
    session['evidence1_list'] = evidence1_list
    session['evidence2_list'] = evidence2_list
    session['processed_evidence1_list'] = processed_evidence1_list
    session['processed_evidence2_list'] = processed_evidence2_list
    session['section'] = section
    # {1: 'relevant evidence', 0: 'non-relevant evidence'}
    pred_evidence1_list = evidenceTrainer.inference(evidence1_list)
    pred_evidence2_list = evidenceTrainer.inference(evidence2_list) if evidence2_list is not None else None
    session['pred_evidence1_list'] = pred_evidence1_list
    session['pred_evidence2_list'] = pred_evidence2_list
    return redirect(url_for('results', section=section, evidence1=evidence1_list, evidence2=evidence2_list))


@app.route("/results")
def results():
    statement = session['statement']
    entail_label = session['entail_label']
    evidence1_list, evidence2_list = session['evidence1_list'], session['evidence2_list']
    pred_evidence1_list, pred_evidence2_list = session['pred_evidence1_list'], session['pred_evidence2_list']
    evidence1, evidence2 = session['evidence1'][:-5], session['evidence2'][:-5]
    evi1 = [e for e in zip(evidence1_list, pred_evidence1_list)]
    evi2 = [e for e in zip(evidence2_list, pred_evidence2_list)] if evidence2_list is not None else evidence2_list
    section = session['section']
    return render_template('results.html', evi1=evi1, evi2=evi2, evidence1=evidence1, evidence2=evidence2, section=section, statement=statement, entail_label=entail_label)


if __name__ == "__main__":
    app.run()
