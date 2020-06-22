import argparse

def cleaning_train():
    parser = argparse.ArgumentParser(description='data cleaning hyper parameter')
    parser.add_argument('--model_mode', type=str, default="bert",
                        help='model_mode')
    parser.add_argument('--max_len', type=int, default=64,
                        help='max_len')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='warmup_ratio')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='num_epochs')
    parser.add_argument('--max_grad_norm', type=int, default=1,
                        help='max_grad_norm')
    parser.add_argument('--log_interval', type=int, default=200,
                        help='log_interval')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='learning_rate')

    args = parser.parse_args()
    return args


def cleaning_test():
    parser = argparse.ArgumentParser(description='data cleaning test hyper parameter')
    parser.add_argument('--model_mode', type=str, default="bert",
                        help='model_mode')
    parser.add_argument('--save_file', type=str, default="cleaning_test.xlsx",
                        help='save file name in cleaning_result')
    parser.add_argument('--model_file', type=str, default="cleaning_model/BERT_27.model",
                        help='load model path')
    parser.add_argument('--data_tsv', type=str, default="data/test_data.txt",
                        help='test data_tsv file')
    parser.add_argument('--data_excel', type=str, default="data/test_data.xlsx",
                        help='test data_excel file')
    parser.add_argument('--max_len', type=int, default=64,
                        help='max_len')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='warmup_ratio')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='num_epochs')
    parser.add_argument('--max_grad_norm', type=int, default=1,
                        help='max_grad_norm')
    parser.add_argument('--log_interval', type=int, default=200,
                        help='log_interval')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='learning_rate')

    args = parser.parse_args()
    return args

def recommend_train():
    parser = argparse.ArgumentParser(description='data cleaning test hyper parameter')
    parser.add_argument('--model_mode', type=str, default="ifm",
                        help='model_mode')
    parser.add_argument('--train_path', type=str, default="data/train_data.xlsx",
                        help='train file path')
    parser.add_argument('--test_path', type=str, default="cleaning_result/cleaning_test_bert.xlsx",
                        help='test file path')
    parser.add_argument('--save_model_file', type=str, default="ifm_model.pickle",
                        help='save file name in recommend_model directory')
    parser.add_argument('--cleaning_train', type=bool, default=True,
                        help='cleaning train dataset')
    parser.add_argument('--cleaning_test', type=bool, default=True,
                        help='cleaning test dataset')
    parser.add_argument('--n_iter', type=int, default=5,
                        help='iteration num of model')

    args = parser.parse_args()
    return args

