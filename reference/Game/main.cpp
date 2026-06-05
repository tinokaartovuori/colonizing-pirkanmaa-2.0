#include "mainwindow.hh"
#include "startdialog.hh"
#include <iostream>

#include <QApplication>

int programMain(int argc, char *argv[])
{
    QApplication app(argc, argv);

    StartDialog dialog;
    MainWindow mainWindow;

    mainWindow.connectDialog(&dialog);

    if (dialog.exec()==StartDialog::Rejected){
        mainWindow.disconnectDialog(&dialog);

        return 0;
    }
    else {
        mainWindow.show();

        return app.exec();
    }
}

int main(int argc, char* argv[])
{
    while (programMain(argc, argv) == 1);

    return 0;
}

