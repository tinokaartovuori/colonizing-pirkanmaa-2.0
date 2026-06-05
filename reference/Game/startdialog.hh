/*
##############################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                                  #
#                                                                            #
# Project: Colonizing Pirkanmaa                                              #
# Program description: Program instructions are located in                   #
#                      Documentation/documentation.pdf                       #
#                                                                            #
# File: startdialog.h                                                        #
#                                                                            #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi                #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi                #
##############################################################################
*/


#ifndef STARTDIALOG_HH
#define STARTDIALOG_HH

#include <QDialog>
#include <stdlib.h>

namespace Ui {
class StartDialog;
}

class StartDialog : public QDialog
{
    Q_OBJECT

signals:
    void sendStartDialogSettings(int, int, int, std::vector<std::string>);

public:
    explicit StartDialog(QWidget *parent = 0);
    ~StartDialog();

private:
    Ui::StartDialog *ui;
    int width_;
    int height_;
    int seed_;
    int playerNum_;

private slots:
    void on_startButton_clicked();
    void on_exitButton_clicked();
    void on_randomizeSeedButton_clicked();

    bool checkCharacters(std::string s);

    void on_widthBox_valueChanged(int width);
    void on_heightBox_valueChanged(int height);
    void on_playersBox_valueChanged(int playerNum);
    void on_seedBox_valueChanged(int seed);
};

#endif // STARTDIALOG_HH
