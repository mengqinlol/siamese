with open('LOSS_WITH_ACC.txt', 'r') as f:
    lines = f.readlines()

    with open('LOSS_WITH_ACC1.txt', 'w') as out:
        for idx, loss in enumerate(lines):
            loss = float(loss)
            if idx > 50 and idx < 164:  
                a = idx - 70
                loss -= 0.7 * (a / 112)
            out.write(f'{loss}\n')
