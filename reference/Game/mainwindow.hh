#ifndef MAINWINDOW_HH
#define MAINWINDOW_HH

#include <QMainWindow>
#include <QGraphicsScene>
#include <startdialog.hh>


#include "DAL/gameeventhandler.h"
#include "DAL/objectmanager.h"
#include "DAL/playermanager.h"
#include "DAL/gamesettingsmanager.h"
#include "DAL/menuobjectmanager.h"

#include "Graphics/gamescene.h"
#include "Graphics/imagevectors.h"
#include "Core/worldgenerator.h"


namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    void connectDialog(StartDialog*);
    void disconnectDialog(StartDialog*);


private slots:
    /*When the start button is clicked on the dialog window,
    this method is called. Parameters are the number of tiles
    in the map grid (width and height). The method also adjusts
    the main window's size so that the map covers the whole
    main window.
    */
    void initializeGame(int width, int height, int seed,
                        std::vector<std::string> playerNames);

    void redrawScene();

    void restart();

private:

    Ui::MainWindow *ui;
    std::shared_ptr<Student::GameScene> gameScene_;
    std::shared_ptr<Student::GameSettingsManager> gameSettingsManager_;
    std::shared_ptr<Student::MenuObjectManager> menuObjectManager_;
    std::shared_ptr<Student::ObjectManager> objectManager_;
    std::shared_ptr<Student::PlayerManager> playerManager_;
    std::shared_ptr<Student::GameEventHandler> eventHandler_;

    int font_id;
};

#endif // MAINWINDOW_HH
