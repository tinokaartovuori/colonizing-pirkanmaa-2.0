#include "startdialog.hh"
#include "ui_startdialog.h"
#include "mainwindow.hh"

#include <algorithm>
#include <cctype>
#include <functional>
#include <iostream>


StartDialog::StartDialog(QWidget *parent) :
    QDialog(parent), ui(new Ui::StartDialog),
    width_(10), height_(10), seed_(1), playerNum_(2)
{
    ui->setupUi(this);
}

StartDialog::~StartDialog()
{
    delete ui;
}

bool StartDialog::checkCharacters(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c){ return std::tolower(c); });

    for (int i = 0; i < (int)s.length(); ++i) {
        char c = s.at(i);
        if(c<'a'|| c>'z') {
            return false;
        }
    }
    return true;
}


void StartDialog::on_startButton_clicked()
{

    std::string playerOne = ui->PlayerOneName->toPlainText().toStdString();
    std::string playerTwo = ui->PlayerTwoName->toPlainText().toStdString();
    std::string playerThree = ui->PlayerThreeName->toPlainText().toStdString();
    std::string playerFour = ui->PlayerFourName->toPlainText().toStdString();
    std::vector<std::string> players = {playerOne, playerTwo};

    if (playerNum_ >= 3) {
        players.push_back(playerThree);
    }
    if (playerNum_ == 4) {
        players.push_back(playerFour);
    }

    for (int i=0;i<playerNum_;++i) {
        std::string name = players.at(i);
        if (!checkCharacters(name)) {
            ui->errorLabel->setText("Error, only letters from a to z accepted.");
            return;
        }
        if (name.length() > 15) {
            ui->errorLabel->setText("Error, name can be only up to 15 charactes long.");
            return;
        }
        if (name == "") {
            ui->errorLabel->setText("Error, enter a name for all players.");
            return;
        }
    }
    emit sendStartDialogSettings(width_, height_, seed_, players);
    emit accept();
    StartDialog::close();

}

void StartDialog::on_exitButton_clicked()
{
    StartDialog::close();
}

void StartDialog::on_randomizeSeedButton_clicked()
{
    srand (time(NULL));
    seed_ = rand() % 200 + 1;
    ui->seedBox->setValue(seed_);
}


void StartDialog::on_widthBox_valueChanged(int width)
{
    width_=width;
}

void StartDialog::on_heightBox_valueChanged(int height)
{
    height_=height;
}

void StartDialog::on_playersBox_valueChanged(int playerNum)
{
    //Enables or disables the name input boxes according to the set player count
    playerNum_=playerNum;
    if (playerNum_==2) {
        ui->PlayerThreeName->setEnabled(0);
        ui->PlayerFourName->setEnabled(0);
    }
    else if (playerNum_==3) {
        ui->PlayerThreeName->setEnabled(1);
        ui->PlayerFourName->setEnabled(0);
    }
    else if (playerNum_==4) {
        ui->PlayerThreeName->setEnabled(1);
        ui->PlayerFourName->setEnabled(1);
    }
}

void StartDialog::on_seedBox_valueChanged(int seed)
{
    seed_=seed;
}
