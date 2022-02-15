from foldrm import *
from datasets import *
from timeit import default_timer as timer
from datetime import timedelta


def main():
    # model, data = acute()
    # model, data = autism()
    # model, data = breastw()
    # model, data = cars()
    # model, data = credit()
    # model, data = heart()
    # model, data = kidney()
    # model, data = krkp()
    # model, data = mushroom()
    # model, data = sonar()
    # model, data = voting()
    model, data = ecoli()
    # model, data = ionosphere()
    # model, data = wine()
    # model, data = adult()
    # model, data = credit_card()
    # model, data = rain()
    # model, data = heloc()
    # model, data = drug()
    # model, data = dry_bean()
    # model, data = eeg()
    # model, data = nursery()
    # model, data = intention()
    # model, data = page_blocks()
    # model, data = parkison()
    # model, data = yeast()
    # model, data = wall_robot()
    # model, data_train, data_test = anneal()
    # model, data_train, data_test = avila()
    # model, data_train, data_test = pendigits()
    # model, data_train, data_test = titanic()

    data_train, data_test = split_data(data, ratio=0.8)

    start = timer()
    model.fit(data_train, ratio=0.5)
    end = timer()

    model.print_asp(simple=True)
    Y = [d[-1] for d in data_test]
    Y_test_hat = model.predict(data_test)
    acc = get_scores(Y_test_hat, data_test)
    print('% acc', round(acc, 4), '# rules', len(model.crs))
    acc, p, r, f1 = scores(Y_test_hat, Y, weighted=True)
    print('% acc', round(acc, 4), 'macro p r f1', round(p, 4), round(r, 4), round(f1, 4), '# rules', len(model.crs))
    print('% foldrm costs: ', timedelta(seconds=end - start), '\n')

    # k = 1
    # for i in range(len(data_test)):
    #     print('Explanation for example number', k, ':')
    #     print(model.explain(data_test[i]))
    #     print('Proof Tree for example number', k, ':')
    #     print(model.proof(data_test[i]))
    #     k += 1


if __name__ == '__main__':
    main()
