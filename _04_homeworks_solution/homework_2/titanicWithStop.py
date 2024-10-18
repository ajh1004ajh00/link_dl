import os
import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from datetime import datetime
import wandb
import argparse
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import sys
from titanic_dataset import TitanicDataset, TitanicTestDataset, get_preprocessed_dataset

def get_data():
    train_dataset, validation_dataset, test_dataset = get_preprocessed_dataset()
    print(len(train_dataset), len(validation_dataset))

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
    validation_data_loader = DataLoader(dataset=validation_dataset, batch_size=len(validation_dataset))
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset))

    return train_data_loader, validation_data_loader, test_data_loader

class MyModel(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(n_input, wandb.config.n_hidden_unit_list[0]),
            nn.ReLU(),
            nn.Linear(wandb.config.n_hidden_unit_list[0], wandb.config.n_hidden_unit_list[1]),
            nn.ReLU(),
            nn.Linear(wandb.config.n_hidden_unit_list[1], n_output),
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
def get_model_and_optimizer():
    my_model = MyModel(n_input=11, n_output=2)  # Titanic 데이터셋에 맞게 n_input=11, n_output=2
    optimizer = optim.SGD(my_model.parameters(), lr=wandb.config.learning_rate)
    return my_model, optimizer

def training_loop(model, optimizer, train_data_loader, validation_data_loader, args):
    n_epochs = wandb.config.epochs
    loss_fn = nn.CrossEntropyLoss()
    best_validation_loss = float('inf')
    patience_counter = 0  # Early stopping을 위한 patience 카운터
    next_print_epoch = 100

    for epoch in range(1, n_epochs + 1):
        model.train()  # 학습 모드 전환
        loss_train = 0.0
        num_trains = 0
        for train_batch in train_data_loader:
            input, target = train_batch['input'], train_batch['target']
            output_train = model(input)
            loss = loss_fn(output_train, target)
            loss_train += loss.item()
            num_trains += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % args.validation_intervals == 0:  # 10번마다 검증 실행
            model.eval()  # 평가 모드 전환
            loss_validation = 0.0
            num_validations = 0
            with torch.no_grad():
                for validation_batch in validation_data_loader:
                    input, target = validation_batch['input'], validation_batch['target']
                    output_validation = model(input)
                    loss = loss_fn(output_validation, target)
                    loss_validation += loss.item()
                    num_validations += 1

            avg_validation_loss = loss_validation / num_validations
            wandb.log({
                "Epoch": epoch,
                "Training loss": loss_train / num_trains,
                "Validation loss": avg_validation_loss
            })

            # Early stopping 조건 확인
            if avg_validation_loss + 0.00001< best_validation_loss:
                best_validation_loss = avg_validation_loss
                patience_counter = 0  # 성능이 개선되면 patience 초기화
            else:
                patience_counter += 1  # 개선되지 않으면 카운터 증가

            if patience_counter >= args.early_stop_patience:  # patience 초과 시 학습 종료
                print(f"Early stopping at epoch {epoch}. Best validation loss: {best_validation_loss:.4f}")
                break
            if epoch >= next_print_epoch:
                print(
                    f"Epoch {epoch}, "
                    f"Training loss {loss_train / num_trains:.4f}, "
                    f"Validation loss {loss_validation / num_validations:.4f}"
                )
                next_print_epoch += 100
        
        #print(f"Epoch {epoch}, Training loss {loss_train / num_trains:.4f}")

def test_and_create_submission(model, test_data_loader):
    model.eval()
    all_predictions = []
    passenger_ids = list(range(892, 892 + len(test_data_loader.dataset)))

    with torch.no_grad():
        for test_batch in test_data_loader:
            input = test_batch['input']
            output = model(input)
            predictions = torch.argmax(output, dim=1).cpu().numpy()
            all_predictions.extend(predictions)

    submission_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': all_predictions
    })
    submission_df.to_csv('submission_test.csv', index=False)
    print("submission_test.csv 파일이 생성되었습니다!")

def main(args):
    current_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': 1e-3,
        'n_hidden_unit_list': [20, 20],
    }

    wandb.init(
        mode="online" if args.wandb else "disabled",
        project="my_model_training",
        notes="Titanic Dataset experiment",
        tags=["my_model", "titanic"],
        name=current_time_str,
        config=config
    )
    print(args)
    print(wandb.config)

    train_data_loader, validation_data_loader, test_data_loader = get_data()

    model, optimizer = get_model_and_optimizer()

    print("#" * 50, 1)

    training_loop(
        model=model,
        optimizer=optimizer,
        train_data_loader=train_data_loader,
        validation_data_loader=validation_data_loader,
        args=args  # args를 전달하여 validation_intervals와 early_stop_patience 사용
    )

    test_and_create_submission(model, test_data_loader)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wandb", action=argparse.BooleanOptionalAction, default=False, help="True or False"
    )

    parser.add_argument(
        "-b", "--batch_size", type=int, default=512, help="Batch size (int, default: 512)"
    )

    parser.add_argument(
        "-e", "--epochs", type=int, default=2_000, help="Number of training epochs (int, default:1_000)"
    )

    parser.add_argument(
        "--validation_intervals", type=int, default=10, help="Interval between validation checks"
    )

    parser.add_argument(
        "--early_stop_patience", type=int, default=20, help="Patience for early stopping"
    )

    args = parser.parse_args()

    main(args)
