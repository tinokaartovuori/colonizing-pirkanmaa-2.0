#include "mainwindow.hh"
#include "ui_mainwindow.h"
#include "iostream"
#include <math.h>
#include <QDebug>
#include <QFontDatabase>



MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow), gameScene_(nullptr),
    objectManager_(nullptr), eventHandler_(nullptr)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::connectDialog(StartDialog* ptr)
{
    QObject::connect(ptr, SIGNAL(sendStartDialogSettings(int, int, int,
                                            std::vector<std::string>)),
                         this, SLOT(initializeGame(int, int, int,
                                            std::vector<std::string>)));

}


void MainWindow::disconnectDialog(StartDialog* ptr)
{
    QObject::disconnect(ptr, SIGNAL(sendStartDialogSettings(int, int, int,
                                            std::vector<std::string>)),
                         this, SLOT(initializeGame(int, int, int,
                                            std::vector<std::string>)));

}


void MainWindow::initializeGame(int width, int height, int seed,
                                std::vector<std::string> playerNames)
{
    font_id = QFontDatabase::addApplicationFont(":/Fonts/PressStart2P.ttf");

    /*Sets on tile length in pixels so that the game window is always
     *around 960 pixels wide. */
    int mapGridSize = round(640 / height / 2) * 2 + round(height*3/2 - 10);

    int menuGridSize = 16;

    int menuGridWidth = 22;
    int menuGridHeight = round(mapGridSize * height / menuGridSize);
    int menuCap = 640 % menuGridHeight;

    if (menuCap > 0) {
        ++menuGridHeight;
    }

    int menuWidth = menuGridSize * menuGridWidth;
    int menuHeight = menuGridSize * menuGridHeight;

    int mapWidth = mapGridSize * width;
    int mapHeight = mapGridSize * height;

    gameSettingsManager_ = std::make_shared<Student::GameSettingsManager>(
                mapGridSize,
                menuGridSize,
                mapWidth,
                mapHeight,
                menuWidth,
                menuHeight
                );

    objectManager_ = std::make_shared<Student::ObjectManager>();
    menuObjectManager_ = std::make_shared<Student::MenuObjectManager>();
    playerManager_ = std::make_shared<Student::PlayerManager>(playerNames, objectManager_);

    eventHandler_ = std::make_shared<Student::GameEventHandler>(
                objectManager_,
                playerManager_,
                menuObjectManager_,
                gameSettingsManager_
                );

    objectManager_->addDALS(
                eventHandler_,
                menuObjectManager_,
                gameSettingsManager_
                );

    menuObjectManager_->addDALS(
                eventHandler_,
                objectManager_,
                playerManager_,
                gameSettingsManager_
                );

    gameScene_ = std::make_shared<Student::GameScene>(nullptr,
                                       eventHandler_,
                                       objectManager_,
                                       menuObjectManager_,
                                       gameSettingsManager_);

    eventHandler_->setGameScene(gameScene_);
    menuObjectManager_->setGameScene(gameScene_);
    objectManager_->setGameScene(gameScene_);

    menuObjectManager_->selectFirstTileMenuView(playerManager_->getCurrentPlayer());


    QObject::connect(gameScene_.get(), SIGNAL(updateScene()),
                         this, SLOT(redrawScene()));


    QObject::connect(eventHandler_.get(),SIGNAL(restartGameSignal()),
                         this, SLOT(restart()));


    /*The main window and graphics view is resized so that the
     *game fills out the window completely. */
    ui->centralwidget->setFixedWidth(mapGridSize*width
                               + menuWidth);
    ui->centralwidget->setFixedHeight(mapGridSize*height);

    ui->graphicsView->setFixedWidth(mapGridSize*width
                               + menuWidth);
    ui->graphicsView->setFixedHeight(mapGridSize*height);

    ui->graphicsView->setHorizontalScrollBarPolicy( Qt::ScrollBarAlwaysOff );
    ui->graphicsView->setVerticalScrollBarPolicy ( Qt::ScrollBarAlwaysOff );

    ui->graphicsView->setSceneRect(QRect(0, 0, mapGridSize*width
                                         + menuWidth, mapGridSize*height ));




    /*Creates a pointer to an object that shows the tile that is
     *below the mouse cursor */
    std::shared_ptr<Student::MouseHoverBorder>
              mousehoverborder = std::make_shared<Student::MouseHoverBorder>
              (Course::Coordinate(0,0), 1, 1,
              eventHandler_, objectManager_);
    objectManager_->setHoverBorder(mousehoverborder);
    mousehoverborder->setImageFiles(ImageVectors::MOUSEHOVERBORDER);
    mousehoverborder->setAnimationOption(AnimationOptions::MOUSEHOVERBORDER);


    Student::WorldGenerator& instance = Student::WorldGenerator::getInstance();
    instance.generateMap(width,
                         height,
                         seed,
                         objectManager_,
                         eventHandler_,
                         gameSettingsManager_,
                         gameScene_);


    //gamescene_->drawItem(defaultView);
    ui->graphicsView->setBackgroundBrush(QColor(53, 119, 44));
    /*Displays the gamescene_*/
    ui->graphicsView->setScene(gameScene_.get());

}

void MainWindow::redrawScene()
{
    ui->graphicsView->viewport()->update();
}

void MainWindow::restart()
{
    if (gameScene_ != nullptr) {
        gameScene_->deleteObjects();
    }

    qApp->exit(1);
}

