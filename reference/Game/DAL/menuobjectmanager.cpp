/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: menuobjectmanager.cpp, see menuobjectmanager.h for more info #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/


#include "menuobjectmanager.h"
#include "iostream"
#include <QDebug>
#include <cmath>



namespace Student {

MenuObjectManager::MenuObjectManager()

{

}

MenuObjectManager::~MenuObjectManager()
{

}

void MenuObjectManager::setGameScene(std::shared_ptr<GameScene> gs)
{
    gameScene_ = gs;
}

void MenuObjectManager::addDALS(
        const std::shared_ptr<Course::iGameEventHandler> gameeventhandler,
        const std::shared_ptr<Student::ObjectManager> objectmanager,
        const std::shared_ptr<Student::PlayerManager> playermanager,
        const std::shared_ptr<Student::GameSettingsManager> gamesettingsmanager)
{
    gameEventHandler_ = gameeventhandler;
    objectManager_ = objectmanager;
    playerManager_ = playermanager;
    gameSettingsManager_ = gamesettingsmanager;
}


void MenuObjectManager::resetMenuView() {

    if (currentMenuView_ != nullptr) {
        gameScene_.lock()->removeContainer(currentMenuView_);
    }

    currentMenuView_ = std::make_shared<Student::MenuView>(
                Course::Coordinate(gameSettingsManager_.lock()->getMapWidth(), 0),
                gameSettingsManager_.lock()->getMenuWidth() / gameSettingsManager_.lock()->getMenuGridSize(),
                gameSettingsManager_.lock()->getMenuHeight() / gameSettingsManager_.lock()->getMenuGridSize(),
                gameSettingsManager_.lock()->getMenuGridSize(),
                gameEventHandler_,
                objectManager_
                );
}

void MenuObjectManager::setPlayerLostMenu(std::vector<std::shared_ptr<Course::PlayerBase>> players, std::vector<std::string> reasons)
{

    resetMenuView();

    int index = 0;
    int yOffset = 10;
    for (auto p : players) {
        std::shared_ptr<Student::MenuObjectContainer> lostPlayer = std::make_shared<Student::MenuObjectContainer>(
                    Course::Coordinate(1, 10 + yOffset * index),
                    20,
                    9,
                    gameSettingsManager_.lock()->getMenuGridSize(),
                    gameEventHandler_,
                    objectManager_
                    );

        lostPlayer->setImageFiles(ImageVectors::MULTI);
        lostPlayer->multiPixMap(true);
        lostPlayer->inverseMultiPixMap(true);

        std::shared_ptr<Student::MenuObjectContainer> colorBall = std::make_shared<Student::MenuObjectContainer>(
                    Course::Coordinate(1, 1),
                    2,
                    2,
                    gameSettingsManager_.lock()->getMenuGridSize(),
                    gameEventHandler_,
                    objectManager_
                    );
        if (p->getPlayerNum() == 1) {
            colorBall->setImageFiles(ImageVectors::RED);
        }
        if (p->getPlayerNum() == 2) {
            colorBall->setImageFiles(ImageVectors::BLUE);
        }
        if (p->getPlayerNum() == 3) {
            colorBall->setImageFiles(ImageVectors::PURPLE);
        }
        if (p->getPlayerNum() == 4) {
            colorBall->setImageFiles(ImageVectors::YELLOW);
        }

        std::shared_ptr<Student::Label> name = std::make_shared<Student::Label>(
                    Course::Coordinate(3, 1),
                    16,
                    9,
                    p->getName() + " lost the game.",
                    12,
                    QColor(200, 200, 200),
                    "LEFT",
                    gameEventHandler_,
                    objectManager_
                    );
        name->setOffset(5);

        std::string reason = "";
        if (reasons.at(index) == "noresources") {
            reason = "Player ran out of resources.";
        }
        if (reasons.at(index) == "conquered") {
            reason = "Players headquarters got conquered.";
        }

        std::shared_ptr<Student::Label> text = std::make_shared<Student::Label>(
                    Course::Coordinate(3, 4),
                    16,
                    9,
                    "<u>Reason:</u><br>" + reason,
                    8,
                    QColor(200, 200, 200),
                    "LEFT",
                    gameEventHandler_,
                    objectManager_
                    );
        name->setOffset(5);

        lostPlayer->addMenuObject(name);
        lostPlayer->addMenuObject(text);
        lostPlayer->addMenuObject(colorBall);
        currentMenuView_->addMenuObject(lostPlayer);
        ++index;
    }


    addOKButton(8, 10 + yOffset * index);

    gameScene_.lock()->drawItem(currentMenuView_);
}

void MenuObjectManager::setWinMenu(std::shared_ptr<Course::PlayerBase> player)
{

    resetMenuView();

    std::shared_ptr<Student::MenuObjectContainer> winnerContainer = std::make_shared<Student::MenuObjectContainer>(
                Course::Coordinate(1, 10),
                20,
                9,
                gameSettingsManager_.lock()->getMenuGridSize(),
                gameEventHandler_,
                objectManager_
                );

    winnerContainer->setImageFiles(ImageVectors::MULTI);
    winnerContainer->multiPixMap(true);
    winnerContainer->inverseMultiPixMap(true);

    std::shared_ptr<Student::MenuObjectContainer> colorBall = std::make_shared<Student::MenuObjectContainer>(
                Course::Coordinate(1, 1),
                2,
                2,
                gameSettingsManager_.lock()->getMenuGridSize(),
                gameEventHandler_,
                objectManager_
                );

    if (player->getPlayerNum() == 1) {
        colorBall->setImageFiles(ImageVectors::RED);
    }
    if (player->getPlayerNum() == 2) {
        colorBall->setImageFiles(ImageVectors::BLUE);
    }
    if (player->getPlayerNum() == 3) {
        colorBall->setImageFiles(ImageVectors::PURPLE);
    }
    if (player->getPlayerNum() == 4) {
        colorBall->setImageFiles(ImageVectors::YELLOW);
    }

    std::shared_ptr<Student::Label> text = std::make_shared<Student::Label>(
                Course::Coordinate(3, 1),
                16,
                9,
                player->getName() + " is the winner!<br><br>Congratulations!",
                12,
                QColor(200, 200, 200),
                "LEFT",
                gameEventHandler_,
                objectManager_
                );
    text->setOffset(5);

    std::shared_ptr<Student::Button> newGame = std::make_shared<Student::Button>(
                    "newGame",
                    Course::Coordinate(3, 11),
                    6,
                    3,
                    "New Game",
                    10,
                    QColor(200, 200, 200),
                    "CENTER",
                    gameEventHandler_,
                    objectManager_
                );

    newGame->setImageFiles(ImageVectors::MULTI);
    newGame->multiPixMap(true);

    std::shared_ptr<Student::Button> quit = std::make_shared<Student::Button>(
                    "quit",
                    Course::Coordinate(11, 11),
                    6,
                    3,
                    "Quit",
                    10,
                    QColor(200, 200, 200),
                    "CENTER",
                    gameEventHandler_,
                    objectManager_
                );

    quit->setImageFiles(ImageVectors::MULTI);
    quit->multiPixMap(true);

    winnerContainer->addMenuObject(text);
    winnerContainer->addMenuObject(colorBall);
    winnerContainer->addMenuObject(newGame);
    winnerContainer->addMenuObject(quit);

    currentMenuView_->addMenuObject(winnerContainer);

    gameScene_.lock()->drawItem(currentMenuView_);
}

void MenuObjectManager::setTieMenu(std::vector<std::shared_ptr<Course::PlayerBase>> players, std::vector<std::string> reasons)
{

    resetMenuView();

    std::shared_ptr<Student::Label> tie_label = std::make_shared<Student::Label>(
                Course::Coordinate(1, 1),
                20,
                3,
                "It is a tie.",
                14,
                QColor(200, 200, 200),
                "CENTER",
                gameEventHandler_,
                objectManager_
                );

    int index = 0;
    int yOffset = 9;
    for (auto p : players) {
        std::shared_ptr<Student::MenuObjectContainer> tiePlayers = std::make_shared<Student::MenuObjectContainer>(
                    Course::Coordinate(1, 4 + yOffset * index),
                    20,
                    8,
                    gameSettingsManager_.lock()->getMenuGridSize(),
                    gameEventHandler_,
                    objectManager_
                    );

        tiePlayers->setImageFiles(ImageVectors::MULTI);
        tiePlayers->multiPixMap(true);
        tiePlayers->inverseMultiPixMap(true);

        std::shared_ptr<Student::MenuObjectContainer> colorBall = std::make_shared<Student::MenuObjectContainer>(
                    Course::Coordinate(1, 1),
                    2,
                    2,
                    gameSettingsManager_.lock()->getMenuGridSize(),
                    gameEventHandler_,
                    objectManager_
                    );
        if (p->getPlayerNum() == 1) {
            colorBall->setImageFiles(ImageVectors::RED);
        }
        if (p->getPlayerNum() == 2) {
            colorBall->setImageFiles(ImageVectors::BLUE);
        }
        if (p->getPlayerNum() == 3) {
            colorBall->setImageFiles(ImageVectors::PURPLE);
        }
        if (p->getPlayerNum() == 4) {
            colorBall->setImageFiles(ImageVectors::YELLOW);
        }

        std::shared_ptr<Student::Label> name = std::make_shared<Student::Label>(
                    Course::Coordinate(3, 1),
                    16,
                    9,
                    p->getName() + " lost the game.",
                    12,
                    QColor(200, 200, 200),
                    "LEFT",
                    gameEventHandler_,
                    objectManager_
                    );
        name->setOffset(5);

        std::string reason = "";
        if (reasons.at(index) == "noresources") {
            reason = "Player ran out of resources.";
        }
        if (reasons.at(index) == "conquered") {
            reason = "Players headquarters got conquered.";
        }

        std::shared_ptr<Student::Label> text = std::make_shared<Student::Label>(
                    Course::Coordinate(3, 4),
                    16,
                    8,
                    "<u>Reason:</u><br>" + reason,
                    8,
                    QColor(200, 200, 200),
                    "LEFT",
                    gameEventHandler_,
                    objectManager_
                    );
        name->setOffset(5);

        tiePlayers->addMenuObject(name);
        tiePlayers->addMenuObject(text);
        tiePlayers->addMenuObject(colorBall);
        currentMenuView_->addMenuObject(tiePlayers);
        ++index;
    }

    std::shared_ptr<Student::Button> newGame = std::make_shared<Student::Button>(
                    "newGame",
                    Course::Coordinate(4, 4 + yOffset * index),
                    6,
                    3,
                    "New Game",
                    10,
                    QColor(200, 200, 200),
                    "CENTER",
                    gameEventHandler_,
                    objectManager_
                );

    newGame->setImageFiles(ImageVectors::MULTI);
    newGame->multiPixMap(true);

    std::shared_ptr<Student::Button> quit = std::make_shared<Student::Button>(
                    "quit",
                    Course::Coordinate(13, 4 + yOffset * index),
                    6,
                    3,
                    "Quit",
                    10,
                    QColor(200, 200, 200),
                    "CENTER",
                    gameEventHandler_,
                    objectManager_
                );

    quit->setImageFiles(ImageVectors::MULTI);
    quit->multiPixMap(true);

    currentMenuView_->addMenuObject(newGame);
    currentMenuView_->addMenuObject(quit);
    currentMenuView_->addMenuObject(tie_label);

    gameScene_.lock()->drawItem(currentMenuView_);
}

void MenuObjectManager::setUnitShopMenuView()
{

    resetMenuView();

    addPlayerTitle();

    addCloseButton();

    addResourceMenu(1, 4);

    std::shared_ptr<Student::MenuObjectContainer> buyUnits = std::make_shared<Student::MenuObjectContainer>(
                Course::Coordinate(1, 12),
                    20,
                    26,
                    gameSettingsManager_.lock()->getMenuGridSize(),
                    gameEventHandler_,
                    objectManager_
                );
    buyUnits->setImageFiles(ImageVectors::MULTI);
    buyUnits->multiPixMap(true);
    buyUnits->inverseMultiPixMap(true);



    std::shared_ptr<Student::Button> addBW = std::make_shared<Student::Button>(
                    "addUnit(BasicWorker)",
                    Course::Coordinate(3, 3),
                    2,
                    3,
                    gameEventHandler_,
                    objectManager_
                );

    QColor bw_buy_text_color;
    if (playerManager_.lock()->getCurrentPlayer()->hasEnoughResources(Course::ConstResourceMaps::BASIC_WORKER_COST) and
            playerManager_.lock()->getCurrentPlayer()->getFreeUnitAmount() > 0) {
        bw_buy_text_color = QColor(200, 200, 200);
    } else {
        bw_buy_text_color = QColor(80, 80, 80);
    }

    std::shared_ptr<Student::Button> addBW_2 = std::make_shared<Student::Button>(
                    "addUnit(BasicWorker)",
                    Course::Coordinate(2, 6),
                    4,
                    2,
                    "BUY",
                    12,
                    bw_buy_text_color,
                    "CENTER",
                    gameEventHandler_,
                    objectManager_
                );
    std::shared_ptr<Student::Button> addExpert = std::make_shared<Student::Button>(
                    "addUnit(Expert)",
                    Course::Coordinate(3, 11),
                    2,
                    3,
                    gameEventHandler_,
                    objectManager_
                );

    QColor ex_buy_text_color;
    if (playerManager_.lock()->getCurrentPlayer()->hasEnoughResources(Course::ConstResourceMaps::EXPERT_COST) and
            playerManager_.lock()->getCurrentPlayer()->getFreeUnitAmount() > 0) {
        ex_buy_text_color = QColor(200, 200, 200);
    } else {
        ex_buy_text_color = QColor(80, 80, 80);
    }
    std::shared_ptr<Student::Button> addExpert_2 = std::make_shared<Student::Button>(
                    "addUnit(Expert)",
                    Course::Coordinate(2, 14),
                    4,
                    2,
                    "BUY",
                    12,
                    ex_buy_text_color,
                    "CENTER",
                    gameEventHandler_,
                    objectManager_
                );
    std::shared_ptr<Student::Button> addSolier = std::make_shared<Student::Button>(
                    "addUnit(Soldier)",
                    Course::Coordinate(3, 19),
                    2,
                    3,
                    gameEventHandler_,
                    objectManager_
                );
    QColor so_buy_text_color;
    if (playerManager_.lock()->getCurrentPlayer()
            ->hasEnoughResources(Course::ConstResourceMaps::SOLDIER_COST) and
            playerManager_.lock()->getCurrentPlayer()->getFreeSoldierAmount() > 0) {
        so_buy_text_color = QColor(200, 200, 200);
    } else {
        so_buy_text_color = QColor(80, 80, 80);
    }
    std::shared_ptr<Student::Button> addSolier_2 = std::make_shared<Student::Button>(
                    "addUnit(Soldier)",
                    Course::Coordinate(2, 22),
                    4,
                    2,
                    "BUY",
                    12,
                    so_buy_text_color,
                    "CENTER",
                    gameEventHandler_,
                    objectManager_
                );

    std::shared_ptr<Student::Label> bw_label = std::make_shared<Student::Label>(
                Course::Coordinate(1,1),
                6,
                2,
                "WORKER",
                12,
                QColor(200, 200, 200),
                "CENTER",
                gameEventHandler_,
                objectManager_
                );
    std::shared_ptr<Student::Label> ex_label = std::make_shared<Student::Label>(
                Course::Coordinate(1,9),
                6,
                2,
                "EXPERT",
                12,
                QColor(200, 200, 200),
                "CENTER",
                gameEventHandler_,
                objectManager_
                );
    std::shared_ptr<Student::Label> sol_label = std::make_shared<Student::Label>(
                Course::Coordinate(1,17),
                6,
                2,
                "SOLDIER",
                12,
                QColor(200, 200, 200),
                "CENTER",
                gameEventHandler_,
                objectManager_
                );

    std::shared_ptr<Student::Label> bw_desc = std::make_shared<Student::Label>(
                Course::Coordinate(7,2),
                12,
                7,
                "<u>Cost:</u> " + std::to_string(abs(Course::ConstResourceMaps::BASIC_WORKER_COST.at(Course::BasicResource::MONEY))) +
                " Coins<br><u>Salary:</u> " + std::to_string(abs(Course::ConstResourceMaps::BASIC_WORKER_SALARY.at(Course::BasicResource::MONEY))) +
                " Coins/round<br><br>A worker can work in mines, farms and power plants. He can also cut down forests.",
                8,
                QColor(200, 200, 200),
                "LEFT",
                gameEventHandler_,
                objectManager_
                );

    std::shared_ptr<Student::Label> ex_desc = std::make_shared<Student::Label>(
                Course::Coordinate(7,10),
                12,
                7,
                "<u>Cost:</u> " + std::to_string(abs(Course::ConstResourceMaps::EXPERT_COST.at(Course::BasicResource::MONEY))) +
                " Coins<br><u>Salary:</u> " + std::to_string(abs(Course::ConstResourceMaps::EXPERT_SALARY.at(Course::BasicResource::MONEY))) +
                " Coins/round<br><br>An expert can work in power plants and mines."
                " He is able to increase efficency a lot.",
                8,
                QColor(200, 200, 200),
                "LEFT",
                gameEventHandler_,
                objectManager_
                );

    std::shared_ptr<Student::Label> so_desc = std::make_shared<Student::Label>(
                Course::Coordinate(7,18),
                12,
                7,
                "<u>Cost:</u><br>" + std::to_string(abs(Course::ConstResourceMaps::SOLDIER_COST.at(Course::BasicResource::MONEY))) +
                " Coins, " + std::to_string(abs(Course::ConstResourceMaps::SOLDIER_COST.at(Course::BasicResource::METAL))) +
                " Metal<br><u>Salary:</u> " + std::to_string(abs(Course::ConstResourceMaps::SOLDIER_SALARY.at(Course::BasicResource::MONEY))) +
                " Coins/round<br><br>A soldier can defend your area and sometimes even conquer tiles from other players.",
                8,
                QColor(200, 200, 200),
                "LEFT",
                gameEventHandler_,
                objectManager_
                );

    addBW->setAnimationOption(AnimationOptions::UNIT);
    addBW->setImageFiles(ImageVectors::BASICWORKER);
    addExpert->setAnimationOption(AnimationOptions::UNIT);
    addExpert->setImageFiles(ImageVectors::EXPERT);
    addSolier->setAnimationOption(AnimationOptions::UNIT);
    addSolier->setImageFiles(ImageVectors::SOLDIER);

    addBW_2->setImageFiles(ImageVectors::MULTI);
    addBW_2->multiPixMap(true);
    addExpert_2->setImageFiles(ImageVectors::MULTI);
    addExpert_2->multiPixMap(true);
    addSolier_2->setImageFiles(ImageVectors::MULTI);
    addSolier_2->multiPixMap(true);


    buyUnits->addMenuObject(addBW);
    buyUnits->addMenuObject(addBW_2);
    buyUnits->addMenuObject(bw_label);

    buyUnits->addMenuObject(bw_desc);
    buyUnits->addMenuObject(ex_desc);
    buyUnits->addMenuObject(so_desc);

    buyUnits->addMenuObject(addExpert);
    buyUnits->addMenuObject(addExpert_2);
    buyUnits->addMenuObject(ex_label);
    buyUnits->addMenuObject(addSolier);
    buyUnits->addMenuObject(addSolier_2);
    buyUnits->addMenuObject(sol_label);


    currentMenuView_->addMenuObject(buyUnits);

    gameScene_.lock()->drawItem(currentMenuView_);
}

void MenuObjectManager::setTileInspectionMenuView(std::shared_ptr<Course::TileBase> tile, int index_for_buildings)
{

    resetMenuView();

    addPlayerTitle();

    addCloseButton();

    std::shared_ptr<Student::MenuObjectContainer> tiledescription = std::make_shared<Student::MenuObjectContainer>(
                Course::Coordinate(1, 4),
                20,
                14,
                gameSettingsManager_.lock()->getMenuGridSize(),
                gameEventHandler_,
                objectManager_
                );

    tiledescription->multiPixMap(true);
    tiledescription->inverseMultiPixMap(true);

    std::shared_ptr<Student::MenuObjectContainer> owner = std::make_shared<Student::MenuObjectContainer>(
                Course::Coordinate(15, 5),
                4,
                2,
                gameSettingsManager_.lock()->getMenuGridSize(),
                gameEventHandler_,
                objectManager_
                );

    if (tile->getOwner() != nullptr) {
        if (tile->getOwner()->getPlayerNum() == 1) {
            owner->setImageFiles(ImageVectors::BAR_RED);
        }
        if (tile->getOwner()->getPlayerNum() == 2) {
            owner->setImageFiles(ImageVectors::BAR_BLUE);
        }
        if (tile->getOwner()->getPlayerNum() == 3) {
            owner->setImageFiles(ImageVectors::BAR_PURPLE);
        }
        if (tile->getOwner()->getPlayerNum() == 4) {
            owner->setImageFiles(ImageVectors::BAR_YELLOW);
        }
    } else {
        owner->setImageFiles(ImageVectors::BAR_NEUTRAL);
    }


    owner->setAnimationOption(AnimationOptions::EMPTY);

    tiledescription->addMenuObject(owner);


    std::shared_ptr<Student::MenuObjectContainer> tilepicture = std::make_shared<Student::MenuObjectContainer>(
                Course::Coordinate(15, 1),
                4,
                4,
                gameSettingsManager_.lock()->getMenuGridSize(),
                gameEventHandler_,
                objectManager_
                );

    tilepicture->setImageFiles(tile->getImageFiles());
    tilepicture->setAnimationOption(tile->getAnimationOption());

    if (tile->getBuilding() != nullptr){
        std::shared_ptr<Student::MenuObjectContainer> buildingpicture = std::make_shared<Student::MenuObjectContainer>(
                    Course::Coordinate(0, 0),
                    4,
                    4,
                    gameSettingsManager_.lock()->getMenuGridSize(),
                    gameEventHandler_,
                    objectManager_
                    );
        buildingpicture->setImageFiles(tile->getBuilding()->getImageFiles());
        buildingpicture->setAnimationOption(tile->getBuilding()->getAnimationOption());
        tilepicture->addMenuObject(buildingpicture);


    }

    std::shared_ptr<Student::MenuObjectContainer> cover_border = std::make_shared<Student::MenuObjectContainer>(
                Course::Coordinate(0, 0),
                4,
                4,
                gameSettingsManager_.lock()->getMenuGridSize(),
                gameEventHandler_,
                objectManager_
                );
    cover_border->setImageFiles(ImageVectors::COVER_BORDER);
    tilepicture->addMenuObject(cover_border);


    std::string tile_name = tile->getType();
    std::string tile_basic_description = "";
    std::shared_ptr<Student::Label> basic_description_label;
    if (tile->getBuilding() != nullptr or tile->getType() == "Forest" or tile->getType() == "Mikontalo" or tile->getType() == "Abundant Forest") {
        if (tile->getType() == "Forest" or tile->getType() == "Mikontalo" or tile->getType() == "Abundant Forest") {
            tile_basic_description = tile->getBasicDescription();
        } else {
            tile_name = tile->getBuilding()->getType();
            if (tile_name == "Nuclear Power Plant") {
                tile_name ="Nuclear PP.";
            }
            if (tile_name == "Hydroelectric Power Plant") {
                tile_name ="Hydroelectric PP.";
            }
            tile_basic_description = tile->getBuilding()->getBasicDescription();

        }
        basic_description_label = std::make_shared<Student::Label>(
                    Course::Coordinate(1,2),
                    13,
                    6,
                    tile_basic_description,
                    8,
                    QColor(200, 200, 200),
                    "LEFT",
                    gameEventHandler_,
                    objectManager_
                    );


        std::string tile_net_description = tile->getNetDescription();
        std::shared_ptr<Student::Label> net_description_label = std::make_shared<Student::Label>(
                    Course::Coordinate(1,8),
                    9,
                    5,
                    tile_net_description,
                    8,
                    QColor(200, 200, 200),
                    "LEFT",
                    gameEventHandler_,
                    objectManager_
                    );


        std::string tile_extra_description = tile->getExtraDescription();
        std::shared_ptr<Student::Label> extra_description_label;
        if ((tile->getType() == "Forest" or tile->getType() == "Mikontalo" or tile->getType() == "Abundant Forest") and tile->getNetDescription() == "") {
            extra_description_label = std::make_shared<Student::Label>(
                        Course::Coordinate(1, 8),
                        9,
                        5,
                        tile_extra_description,
                        8,
                        QColor(200, 200, 200),
                        "LEFT",
                        gameEventHandler_,
                        objectManager_
                        );
        } else {
            extra_description_label = std::make_shared<Student::Label>(
                        Course::Coordinate(10, 8),
                        9,
                        5,
                        tile_extra_description,
                        8,
                        QColor(200, 200, 200),
                        "LEFT",
                        gameEventHandler_,
                        objectManager_
                        );
        }

        extra_description_label->setMargin(12);
        net_description_label->setMargin(12);

        if (tile->getExtraDescription() != "" or tile->getNetDescription() != "") {
            std::shared_ptr<Student::MenuObjectContainer> extra_bg = std::make_shared<Student::MenuObjectContainer>(
                        Course::Coordinate(1, 8),
                        18,
                        5,
                        gameSettingsManager_.lock()->getMenuGridSize(),
                        gameEventHandler_,
                        objectManager_
                        );

            extra_bg->multiPixMap(true);
            extra_bg->setImageFiles(ImageVectors::MULTI);
            extra_bg->inverseMultiPixMap(true);
            tiledescription->addMenuObject(extra_bg);
        }


        tiledescription->addMenuObject(extra_description_label);
        tiledescription->addMenuObject(net_description_label);


    } else {
        tile_basic_description = tile->getBasicDescription();

        basic_description_label = std::make_shared<Student::Label>(
                    Course::Coordinate(1,2),
                    14,
                    10,
                    tile_basic_description,
                    8,
                    QColor(200, 200, 200),
                    "LEFT",
                    gameEventHandler_,
                    objectManager_
                    );

    }
    std::shared_ptr<Student::Label> tile_title = std::make_shared<Student::Label>(
                Course::Coordinate(1,0),
                14,
                2,
                "<u>" + tile_name + "</u>",
                12,
                QColor(200, 200, 200),
                "LEFT-CENTER",
                gameEventHandler_,
                objectManager_
                );

    tile_title->setOffset(5);

    tiledescription->setImageFiles(ImageVectors::MULTI);
    tiledescription->addMenuObject(tile_title);
    tiledescription->addMenuObject(basic_description_label);


    std::shared_ptr<Student::MenuObjectContainer> units_view = std::make_shared<Student::MenuObjectContainer>(
                Course::Coordinate(1, 19),
                20,
                9,
                gameSettingsManager_.lock()->getMenuGridSize(),
                gameEventHandler_,
                objectManager_
                );

    units_view->multiPixMap(true);
    units_view->inverseMultiPixMap(true);
    units_view->setImageFiles(ImageVectors::MULTI);

    if (tile->getOwner() == playerManager_.lock()->getCurrentPlayer()) {

        if (tile->getUnitCount() > 0) {
            currentMenuView_->addMenuObject(units_view);
        }

        if (tile->getUnitCount() > 0) {
            int index = 0;
            for (auto unit : tile->getUnits()) {
                std::shared_ptr<Student::MenuObjectContainer> unit_cell = std::make_shared<Student::MenuObjectContainer>(
                            Course::Coordinate(3 + index + 4*index, 1),
                            6,
                            8,
                            gameSettingsManager_.lock()->getMenuGridSize(),
                            gameEventHandler_,
                            objectManager_
                            );

                std::vector<std::string> img_ = unit->getImageFiles();
                AnimationOption ani_ = unit->getAnimationOption();

                std::shared_ptr<Student::Button> unit_pic = std::make_shared<Student::Button>(
                                "moveUnit(" + std::to_string(index) + ")",
                                Course::Coordinate(1, 0),
                                2,
                                3,
                                gameEventHandler_,
                                objectManager_
                            );
                unit_pic->setImageFiles(img_);
                unit_pic->setAnimationOption(ani_);

                std::shared_ptr<Student::Button> move_unit = std::make_shared<Student::Button>(
                                "moveUnit(" + std::to_string(index) + ")",
                                Course::Coordinate(0, 3),
                                4,
                                2,
                                "MOVE",
                                8,
                                QColor(200, 200, 200),
                                "CENTER",
                                gameEventHandler_,
                                objectManager_
                            );

                unit_pic->setCorrespondingTile(tile);
                move_unit->setCorrespondingTile(tile);

                move_unit->setImageFiles(ImageVectors::MULTI);
                move_unit->multiPixMap(true);

                std::shared_ptr<Student::Button> del_unit = std::make_shared<Student::Button>(
                                "delUnit(" + std::to_string(index) + ")",
                                Course::Coordinate(0, 5),
                                4,
                                2,
                                "DEL",
                                8,
                                QColor(200, 200, 200),
                                "CENTER",
                                gameEventHandler_,
                                objectManager_
                            );

                del_unit->setCorrespondingTile(tile);

                del_unit->setImageFiles(ImageVectors::MULTI);
                del_unit->multiPixMap(true);

                unit_cell->addMenuObject(unit_pic);
                unit_cell->addMenuObject(move_unit);
                unit_cell->addMenuObject(del_unit);

                units_view->addMenuObject(unit_cell);

                ++index;

            }

        }

        int y_for_builds;
        if (tile->getUnitCount() > 0 and !tile->getBuildableBuildings().empty()) {
            y_for_builds = 29;
        } else {
            y_for_builds = 19;
        }


        std::shared_ptr<Student::MenuObjectContainer> building_view = std::make_shared<Student::MenuObjectContainer>(
                    Course::Coordinate(1, y_for_builds),
                    20,
                    13,
                    gameSettingsManager_.lock()->getMenuGridSize(),
                    gameEventHandler_,
                    objectManager_
                    );

        if (tile->getBuilding() == nullptr and !tile->getBuildableBuildings().empty()) {
            currentMenuView_->addMenuObject(building_view);
        }

        if (index_for_buildings < 0) {
            index_for_buildings = tile->getBuildableBuildings().size() - 1;
        }
        else if (index_for_buildings > (int)tile->getBuildableBuildings().size() - 1) {
            index_for_buildings = 0;
        }
        if (!tile->getBuildableBuildings().empty()) {
            std::string building_type =
                         tile->getBuildableBuildings().at(index_for_buildings);

            std::string name;
            std::vector<std::string> imagevector;
            std::vector<std::string> image_bg = tile->getImageFiles();

            if (tile->getType() == "Forest") {
                image_bg = ImageVectors::GRASSLAND;
            }

            AnimationOption animationoption;
            AnimationOption animationoption_bg = tile->getAnimationOption();

            Course::ResourceMap resources_needed;
            Course::ResourceMap production;
            std::string description_for_building;
            std::string extra_production_text = "";

            if (building_type == "Farm") {
                name = building_type;
                imagevector = ImageVectors::FARM;
                animationoption = AnimationOptions::EMPTY;
                resources_needed = Course::ConstResourceMaps::FARM_BUILD_COST;
                production = Course::ConstResourceMaps::FARM_PRODUCTION;
                description_for_building = ConstDescriptionMaps::FARM_SHOP_DESCRIPTION;
            }
            else if (building_type == "Bridge") {
                name = building_type;
                int orientation = 0;
                if (std::dynamic_pointer_cast<Student::River>(tile) != nullptr) {
                    orientation = std::dynamic_pointer_cast<Student::River>(tile)->getRiverOrientation();
                }
                if (orientation == 0) {
                    imagevector = ImageVectors::BRIDGENS;
                }
                if (orientation == 1) {
                    imagevector = ImageVectors::BRIDGEWE;
                }
                animationoption = AnimationOptions::EMPTY;
                resources_needed = Course::ConstResourceMaps::BRIDGE_BUILD_COST;
                production = Course::ConstResourceMaps::BRIDGE_PRODUCTION;
                description_for_building = ConstDescriptionMaps::BRIDGE_SHOP_DESCRIPTION;
            }
            else if (building_type == "Hydroelectric Power Plant") {
                name = "Hydroelectric PP.";

                int orientation = 0;
                if (std::dynamic_pointer_cast<Student::River>(tile) != nullptr) {
                    orientation = std::dynamic_pointer_cast<Student::River>(tile)->getRiverOrientation();
                }
                if (orientation == 0) {
                    imagevector = ImageVectors::HYDROPOWERWE;
                }
                if (orientation == 1) {
                    imagevector = ImageVectors::HYDROPOWERNS;
                }
                animationoption = AnimationOptions::HEPP;
                resources_needed = Course::ConstResourceMaps::HEPP_BUILD_COST;
                production = Course::ConstResourceMaps::HEPP_PRODUCTION;
                description_for_building = ConstDescriptionMaps::HEPP_SHOP_DESCRIPTION;
                extra_production_text = "(for each unit)";
            }
            else if (building_type == "Mine") {
                name = building_type;
                imagevector = ImageVectors::MINE;
                animationoption = AnimationOptions::EMPTY;
                resources_needed = Course::ConstResourceMaps::MINE_BUILD_COST;
                production = Course::ConstResourceMaps::MINE_PRODUCTION;
                description_for_building = ConstDescriptionMaps::MINE_SHOP_DESCRIPTION;
                extra_production_text = "(for each unit)";
            }
            else if (building_type == "Nuclear Power Plant") {
                name = "Nuclear PP.";
                imagevector = ImageVectors::NUCLEARPLANT;
                animationoption = AnimationOptions::NUCLEAR;
                resources_needed = Course::ConstResourceMaps::NUCLEARPP_BUILD_COST;
                production = Course::ConstResourceMaps::NUCLEARPP_PRODUCTION;
                description_for_building = ConstDescriptionMaps::NUCLEAR_SHOP_DESCRIPTION;
                extra_production_text = "(for each unit)";
            }
            else if (building_type == "Outpost") {
                name = building_type;
                imagevector = ImageVectors::OUTPOST;
                animationoption = AnimationOptions::OUTPOST;
                resources_needed = Course::ConstResourceMaps::OUTPOST_BUILD_COST;
                production = Course::ConstResourceMaps::OUTPOST_PRODUCTION;
                description_for_building = ConstDescriptionMaps::OUTPOST_SHOP_DESCRIPTION;
            }
            else if (building_type == "Village") {
                name = building_type;
                imagevector = ImageVectors::VILLAGE;
                animationoption = AnimationOptions::EMPTY;
                resources_needed = Course::ConstResourceMaps::VILLAGE_BUILD_COST;
                production = Course::ConstResourceMaps::VILLAGE_PRODUCTION;
                description_for_building = ConstDescriptionMaps::VILLAGE_SHOP_DESCRIPTION;
            }

            QColor switch_building = QColor(200, 200, 200);
            if (tile->getBuildableBuildings().size() == 1) {
                switch_building = QColor(80, 80, 80);
            }

            std::shared_ptr<Student::Button> previous_building = std::make_shared<Student::Button>(
                            "switchBuyMenu",
                            Course::Coordinate(1, 1),
                            3,
                            2,
                            "&lt;",
                            10,
                            switch_building,
                            "CENTER",
                            gameEventHandler_,
                            objectManager_
                        );

            previous_building->setCorrespondingTile(tile);
            previous_building->setHoldingIndex(index_for_buildings - 1);
            previous_building->setImageFiles(ImageVectors::MULTI);
            previous_building->multiPixMap(true);

            std::shared_ptr<Student::Button> next_building = std::make_shared<Student::Button>(
                            "switchBuyMenu",
                            Course::Coordinate(16, 1),
                            3,
                            2,
                            "&gt;",
                            10,
                            switch_building,
                            "CENTER",
                            gameEventHandler_,
                            objectManager_
                        );

            next_building->setHoldingIndex(index_for_buildings + 1);
            next_building->setCorrespondingTile(tile);
            next_building->setImageFiles(ImageVectors::MULTI);
            next_building->multiPixMap(true);

            std::shared_ptr<Student::Label> building_title = std::make_shared<Student::Label>(
                        Course::Coordinate(4, 1),
                        12,
                        2,
                        name,
                        10,
                        QColor(200, 200, 200),
                        "CENTER",
                        gameEventHandler_,
                        objectManager_
                        );

            std::shared_ptr<Student::MenuObjectContainer> title_bg = std::make_shared<Student::MenuObjectContainer>(
                        Course::Coordinate(4, 1),
                        12,
                        2,
                        gameSettingsManager_.lock()->getMenuGridSize(),
                        gameEventHandler_,
                        objectManager_
                        );

            title_bg->setImageFiles(ImageVectors::MULTI);
            title_bg->multiPixMap(true);


            std::shared_ptr<Student::MenuObjectContainer> building_picture_bg = std::make_shared<Student::MenuObjectContainer>(
                        Course::Coordinate(1, 4),
                        4,
                        4,
                        gameSettingsManager_.lock()->getMenuGridSize(),
                        gameEventHandler_,
                        objectManager_
                        );

            building_picture_bg->setImageFiles(image_bg);
            building_picture_bg->setAnimationOption(animationoption_bg);



            std::shared_ptr<Student::MenuObjectContainer> building_picture = std::make_shared<Student::MenuObjectContainer>(
                        Course::Coordinate(0, 0),
                        4,
                        4,
                        gameSettingsManager_.lock()->getMenuGridSize(),
                        gameEventHandler_,
                        objectManager_
                        );

            std::shared_ptr<Student::MenuObjectContainer> cover_border = std::make_shared<Student::MenuObjectContainer>(
                        Course::Coordinate(0, 0),
                        4,
                        4,
                        gameSettingsManager_.lock()->getMenuGridSize(),
                        gameEventHandler_,
                        objectManager_
                        );

            cover_border->setImageFiles(ImageVectors::COVER_BORDER);

            std::string build_cost_string = "<u>Cost:</u><br>";
            std::string build_production_string  = "<u>Products:</u><br>";

            for (auto const& c : resources_needed)
            {
                bool hasEnoughResource = true;
                const Course::ResourceMap rm = {
                    {c.first, c.second},
                };

                if (!playerManager_.lock()->getCurrentPlayer()->hasEnoughResources(rm)) {
                    hasEnoughResource = false;
                }
                std::string material;
                if (c.first == Course::BasicResource::MONEY) {
                    material = "Money";
                }
                if (c.first == Course::BasicResource::WOOD) {
                    material = "Wood";
                }
                if (c.first == Course::BasicResource::STONE) {
                    material = "Stone";
                }
                if (c.first == Course::BasicResource::METAL) {
                    material = "Metal";
                }
                std::string extra_color = "color: rgb(200, 200, 200)";
                if (!hasEnoughResource) {
                    extra_color = "color: rgb(255, 50, 50)";
                }

                build_cost_string += "<span style='" + extra_color + "'>" + std::to_string(c.second * (-1)) + " " + material + "</span><br>";
            }

            for (auto const& p : production)
            {
                std::string material;
                if (p.first == Course::BasicResource::MONEY) {
                    material = "Money";
                }
                if (p.first == Course::BasicResource::WOOD) {
                    material = "Wood";
                }
                if (p.first == Course::BasicResource::STONE) {
                    material = "Stone";
                }
                if (p.first == Course::BasicResource::METAL) {
                    material = "Metal";
                }
                if (material == "Money" and building_type == "Farm") {
                    build_production_string += std::to_string(p.second) +
                            " " + material + " every 4 rounds<br>";
                }else {
                    build_production_string += std::to_string(p.second) +
                            " " + material + "/r<br>";
                }

            }
            build_production_string += extra_production_text;

            std::shared_ptr<Student::Label> data = std::make_shared<Student::Label>(
                        Course::Coordinate(5, 4),
                        6,
                        8,
                        "<p style='line-height: 145%; color: rgb(200, 200, 200)'>"
                        + build_cost_string + build_production_string + "</p>",
                        8,
                        QColor(200, 200, 200),
                        "LEFT",
                        gameEventHandler_,
                        objectManager_
                        );
            data->setMargin(8);

            std::shared_ptr<Student::MenuObjectContainer> data_bg = std::make_shared<Student::MenuObjectContainer>(
                        Course::Coordinate(5, 4),
                        14,
                        8,
                        gameSettingsManager_.lock()->getMenuGridSize(),
                        gameEventHandler_,
                        objectManager_
                        );

            data_bg->setImageFiles(ImageVectors::MULTI);
            data_bg->multiPixMap(true);
            data_bg->inverseMultiPixMap(true);
            data->setNoRightMargin(true);

            std::shared_ptr<Student::Label> building_desc = std::make_shared<Student::Label>(
                        Course::Coordinate(11, 4),
                        8,
                        8,
                        description_for_building,
                        8,
                        QColor(200, 200, 200),
                        "LEFT",
                        gameEventHandler_,
                        objectManager_
                        );
            building_desc->setMargin(8);

            std::shared_ptr<Student::MenuObjectContainer> build_desc_bg = std::make_shared<Student::MenuObjectContainer>(
                        Course::Coordinate(11, 4),
                        8,
                        8,
                        gameSettingsManager_.lock()->getMenuGridSize(),
                        gameEventHandler_,
                        objectManager_
                        );

            QColor buy_text_color;
            if (playerManager_.lock()->getCurrentPlayer()->hasEnoughResources(resources_needed)) {
                buy_text_color = QColor(200, 200, 200);
            } else {
                buy_text_color = QColor(80, 80, 80);
            }

            std::shared_ptr<Student::Button> buy_building = std::make_shared<Student::Button>(
                            "build(" + building_type + ")",
                            Course::Coordinate(1, 8),
                            4,
                            4,
                            "BUY",
                            12,
                            buy_text_color,
                            "CENTER",
                            gameEventHandler_,
                            objectManager_
                        );

            buy_building->setImageFiles(ImageVectors::MULTI);
            buy_building->multiPixMap(true);
            buy_building->setCorrespondingTile(tile);

            build_desc_bg->setImageFiles(ImageVectors::MULTI);
            build_desc_bg->multiPixMap(true);
            build_desc_bg->inverseMultiPixMap(true);

            building_picture->setImageFiles(imagevector);
            building_picture->setAnimationOption(animationoption);

            building_view->addMenuObject(building_picture_bg);
            building_picture_bg->addMenuObject(building_picture);
            building_picture_bg->addMenuObject(cover_border);

            building_view->addMenuObject(next_building);
            building_view->addMenuObject(previous_building);

            building_view->addMenuObject(title_bg);
            building_view->addMenuObject(building_title);

            building_view->addMenuObject(data_bg);
            //building_view->addMenuObject(build_desc_bg);

            building_view->addMenuObject(data);
            building_view->addMenuObject(building_desc);

            building_view->addMenuObject(buy_building);

            building_view->multiPixMap(true);
            building_view->inverseMultiPixMap(true);
            building_view->setImageFiles(ImageVectors::MULTI);
        }

    } else {

        if (tile->getConqueringUnitCount() > 0) {
            currentMenuView_->addMenuObject(units_view);
        }

        if (tile->getConqueringUnitCount() > 0) {
            int index = 0;
            for (auto unit : tile->getConqueringUnits()) {
                std::shared_ptr<Student::MenuObjectContainer> unit_cell = std::make_shared<Student::MenuObjectContainer>(
                            Course::Coordinate(3 + index + 4*index, 1),
                            6,
                            8,
                            gameSettingsManager_.lock()->getMenuGridSize(),
                            gameEventHandler_,
                            objectManager_
                            );

                std::vector<std::string> img_ = unit->getImageFiles();
                AnimationOption ani_ = unit->getAnimationOption();

                std::shared_ptr<Student::Button> unit_pic = std::make_shared<Student::Button>(
                                "moveUnit(" + std::to_string(index) + ", enemy)",
                                Course::Coordinate(1, 0),
                                2,
                                3,
                                gameEventHandler_,
                                objectManager_
                            );

                unit_pic->setCorrespondingTile(tile);

                unit_pic->setImageFiles(img_);
                unit_pic->setAnimationOption(ani_);

                std::shared_ptr<Student::Button> move_unit = std::make_shared<Student::Button>(
                                "moveUnit(" + std::to_string(index) + ", enemy)",
                                Course::Coordinate(0, 3),
                                4,
                                2,
                                "MOVE",
                                8,
                                QColor(200, 200, 200),
                                "CENTER",
                                gameEventHandler_,
                                objectManager_
                            );

                move_unit->setCorrespondingTile(tile);

                move_unit->setImageFiles(ImageVectors::MULTI);
                move_unit->multiPixMap(true);

                std::shared_ptr<Student::Button> del_unit = std::make_shared<Student::Button>(
                                "delUnit(" + std::to_string(index) + ", enemy)",
                                Course::Coordinate(0, 5),
                                4,
                                2,
                                "DEL",
                                8,
                                QColor(200, 200, 200),
                                "CENTER",
                                gameEventHandler_,
                                objectManager_
                            );

                del_unit->setCorrespondingTile(tile);

                del_unit->setImageFiles(ImageVectors::MULTI);
                del_unit->multiPixMap(true);

                unit_cell->addMenuObject(unit_pic);
                unit_cell->addMenuObject(move_unit);
                unit_cell->addMenuObject(del_unit);

                units_view->addMenuObject(unit_cell);

                ++index;

            }

        }

        if (tile->getUnitCount() > 0) {
            int enemy_y = 19;
            if (tile->getConqueringUnitCount() > 0) {
                enemy_y = 29;
            }
            std::shared_ptr<Student::MenuObjectContainer> enemy_units_view = std::make_shared<Student::MenuObjectContainer>(
                        Course::Coordinate(1, enemy_y),
                        20,
                        5,
                        gameSettingsManager_.lock()->getMenuGridSize(),
                        gameEventHandler_,
                        objectManager_
                        );

            enemy_units_view->multiPixMap(true);
            enemy_units_view->inverseMultiPixMap(true);
            enemy_units_view->setImageFiles(ImageVectors::MULTI);

            if (tile->getUnitCount() > 0) {
                currentMenuView_->addMenuObject(enemy_units_view);
            }

            int index = 0;
            for (auto unit : tile->getUnits()) {
                std::shared_ptr<Student::MenuObjectContainer> unit_cell = std::make_shared<Student::MenuObjectContainer>(
                            Course::Coordinate(3 + index + 4*index, 1),
                            6,
                            8,
                            gameSettingsManager_.lock()->getMenuGridSize(),
                            gameEventHandler_,
                            objectManager_
                            );

                std::vector<std::string> img_ = unit->getImageFiles();
                AnimationOption ani_ = unit->getAnimationOption();

                std::shared_ptr<Student::Button> unit_pic = std::make_shared<Student::Button>(
                                "none",
                                Course::Coordinate(1, 0),
                                2,
                                3,
                                gameEventHandler_,
                                objectManager_
                            );
                unit_pic->setImageFiles(img_);
                unit_pic->setAnimationOption(ani_);

                unit_cell->addMenuObject(unit_pic);
                enemy_units_view->addMenuObject(unit_cell);

                ++index;

            }
        }


    }

    currentMenuView_->setImageFiles(ImageVectors::MENU);
    currentMenuView_->setAnimationOption(AnimationOptions::MENU);

    currentMenuView_->addMenuObject(tiledescription);

    tiledescription->addMenuObject(tilepicture);
    gameScene_.lock()->drawItem(currentMenuView_);

}

void MenuObjectManager::selectFirstTileMenuView(std::shared_ptr<Course::PlayerBase> player) {

    std::string name = player->getName();

    resetMenuView();
    addPlayerTitle();

    std::shared_ptr<Student::MenuObjectContainer> cont = std::make_shared<Student::MenuObjectContainer>(
                Course::Coordinate(1, 4),
                20,
                16,
                gameSettingsManager_.lock()->getMenuGridSize(),
                gameEventHandler_,
                objectManager_
                );

    cont->setImageFiles(ImageVectors::MULTI);
    cont->multiPixMap(true);
    cont->inverseMultiPixMap(true);

    std::shared_ptr<Student::Label> instruction = std::make_shared<Student::Label>(
                Course::Coordinate(1,1),
                18,
                12,
                name + " choose your starting tile! Starting tile must be a grassland. "
                       "You will also get all of the other tiles next to the chosen tile. "
                       "Choose carefully.<br><br>Good luck!",
                12,
                QColor(200, 200, 200),
                "LEFT",
                gameEventHandler_,
                objectManager_
                );

    cont->addMenuObject(instruction);
    currentMenuView_->addMenuObject(cont);

    gameScene_.lock()->drawItem(currentMenuView_);

}

void MenuObjectManager::setDefaultMenuView()
{
    resetMenuView();

    addPlayerTitle();

    addResourceMenu(1, 4);

    addUnitMenu(1, 28);

    std::shared_ptr<Student::Button> buymenu = std::make_shared<Student::Button>(
                "openbuymenu",
                Course::Coordinate(1, 12),
                6,
                3,
                "UNIT SHOP",
                10,
                QColor(200, 200, 200),
                "CENTER",
                gameEventHandler_,
                objectManager_
            );

    std::shared_ptr<Student::Button> statsmenu = std::make_shared<Student::Button>(
                "openstatsmenu",
                Course::Coordinate(8, 12),
                6,
                3,
                "STATS",
                10,
                QColor(200, 200, 200),
                "CENTER",
                gameEventHandler_,
                objectManager_
            );

    std::shared_ptr<Student::Button> helpmenu = std::make_shared<Student::Button>(
                "help",
                Course::Coordinate(15, 12),
                6,
                3,
                "HELP",
                10,
                QColor(200, 200, 200),
                "CENTER",
                gameEventHandler_,
                objectManager_
            );

    std::shared_ptr<Student::Button> endturn = std::make_shared<Student::Button>(
                "endturn",
                Course::Coordinate(1, 38),
                20,
                4,
                "END TURN",
                10,
                QColor(200, 200, 200),
                "CENTER",
                gameEventHandler_,
                objectManager_
            );


    addNetMenu(1, 16);

    buymenu->multiPixMap(true);
    buymenu->setImageFiles(ImageVectors::MULTI);

    endturn->multiPixMap(true);
    statsmenu->multiPixMap(true);
    helpmenu->multiPixMap(true);
    endturn->setImageFiles(ImageVectors::MULTI);
    statsmenu->setImageFiles(ImageVectors::MULTI);
    helpmenu->setImageFiles(ImageVectors::MULTI);

    currentMenuView_->addMenuObject(buymenu);
    currentMenuView_->addMenuObject(endturn);
    currentMenuView_->addMenuObject(statsmenu);
    currentMenuView_->addMenuObject(helpmenu);

    gameScene_.lock()->drawItem(currentMenuView_);

}

void MenuObjectManager::setStatMenuView() {

    resetMenuView();

    addPlayerTitle();

    addCloseButton();

    addRoundsPlayed(1, 4);

    addTilePercents(1, 9);

    gameScene_.lock()->drawItem(currentMenuView_);
}

void MenuObjectManager::addNetMenu(int x, int y) {

    std::shared_ptr<Student::MenuObjectContainer> netValueMenu = std::make_shared<Student::MenuObjectContainer>(
                Course::Coordinate(x, y),
                    20,
                    11,
                    gameSettingsManager_.lock()->getMenuGridSize(),
                    gameEventHandler_,
                    objectManager_
                );

    netValueMenu->multiPixMap(true);
    netValueMenu->inverseMultiPixMap(true);
    netValueMenu->setImageFiles(ImageVectors::MULTI);

    std::shared_ptr<Student::MenuObjectContainer> money_icon = std::make_shared<Student::MenuObjectContainer>(
                Course::Coordinate(1, 2),
                    2,
                    2,
                    gameSettingsManager_.lock()->getMenuGridSize(),
                    gameEventHandler_,
                    objectManager_
                );
    money_icon->setImageFiles(ImageVectors::MONEY);
    std::shared_ptr<Student::MenuObjectContainer> wood_icon = std::make_shared<Student::MenuObjectContainer>(
                Course::Coordinate(1, 4),
                    2,
                    2,
                    gameSettingsManager_.lock()->getMenuGridSize(),
                    gameEventHandler_,
                    objectManager_
                );
    wood_icon->setImageFiles(ImageVectors::WOOD);
    std::shared_ptr<Student::MenuObjectContainer> stone_icon = std::make_shared<Student::MenuObjectContainer>(
                Course::Coordinate(1, 6),
                    2,
                    2,
                    gameSettingsManager_.lock()->getMenuGridSize(),
                    gameEventHandler_,
                    objectManager_
                );
    stone_icon->setImageFiles(ImageVectors::STONE);
    std::shared_ptr<Student::MenuObjectContainer> metal_icon = std::make_shared<Student::MenuObjectContainer>(
                Course::Coordinate(1, 8),
                    2,
                    2,
                    gameSettingsManager_.lock()->getMenuGridSize(),
                    gameEventHandler_,
                    objectManager_
                );
    metal_icon->setImageFiles(ImageVectors::METAL);

    netValueMenu->addMenuObject(money_icon);
    netValueMenu->addMenuObject(wood_icon);
    netValueMenu->addMenuObject(stone_icon);
    netValueMenu->addMenuObject(metal_icon);

    std::shared_ptr<Student::Label> revenue_label = std::make_shared<Student::Label>(
                Course::Coordinate(3,0),
                5,
                2,
                "Revenue",
                8,
                QColor(200, 200, 200),
                "CENTER",
                gameEventHandler_,
                objectManager_
                );
    std::shared_ptr<Student::Label> expenses_label = std::make_shared<Student::Label>(
                Course::Coordinate(8,0),
                5,
                2,
                "Expenses",
                8,
                QColor(200, 200, 200),
                "CENTER",
                gameEventHandler_,
                objectManager_
                );
    std::shared_ptr<Student::Label> net_label = std::make_shared<Student::Label>(
                Course::Coordinate(13,0),
                6,
                2,
                "<u>Net</u>",
                10,
                QColor(200, 200, 200),
                "CENTER",
                gameEventHandler_,
                objectManager_
                );


    Course::ResourceMap revenue = gameEventHandler_.lock()->getCurrentRevenue();
    Course::ResourceMap expenses = gameEventHandler_.lock()->getCurrentExpences();
    Course::ResourceMap net = gameEventHandler_.lock()->getCurrentNet();

    std::string revenue_money_text = "-";
    if (revenue.count(Course::BasicResource::MONEY) > 0) {
        if (revenue.at(Course::BasicResource::MONEY) != 0) {
            revenue_money_text = std::to_string(revenue[Course::BasicResource::MONEY]);
        }
    }
    std::shared_ptr<Student::Label> revenue_money_label = std::make_shared<Student::Label>(
                Course::Coordinate(3, 2),
                5,
                2,
                revenue_money_text,
                10,
                QColor(200, 200, 200),
                "CENTER",
                gameEventHandler_,
                objectManager_
                );

    std::string expenses_money_text = "-";
    if (expenses.count(Course::BasicResource::MONEY) > 0) {
        if (expenses.at(Course::BasicResource::MONEY) != 0) {
            expenses_money_text = std::to_string(expenses[Course::BasicResource::MONEY]);
        }
    }
    std::shared_ptr<Student::Label> expenses_money_label = std::make_shared<Student::Label>(
                Course::Coordinate(8, 2),
                5,
                2,
                expenses_money_text,
                10,
                QColor(200, 200, 200),
                "CENTER",
                gameEventHandler_,
                objectManager_
                );

    std::string net_money_text = "-";
    if (net.count(Course::BasicResource::MONEY) > 0) {
        if (net.at(Course::BasicResource::MONEY) != 0) {
            net_money_text = std::to_string(net[Course::BasicResource::MONEY]);
        }
    }
    std::shared_ptr<Student::Label> net_money_label = std::make_shared<Student::Label>(
                Course::Coordinate(13, 2),
                6,
                2,
                net_money_text,
                10,
                QColor(200, 200, 200),
                "CENTER",
                gameEventHandler_,
                objectManager_
                );

    netValueMenu->addMenuObject(revenue_money_label);
    netValueMenu->addMenuObject(expenses_money_label);
    netValueMenu->addMenuObject(net_money_label);

    std::string revenue_wood_text = "-";
    if (revenue.count(Course::BasicResource::WOOD) > 0) {
        if (revenue.at(Course::BasicResource::WOOD) != 0) {
            revenue_wood_text = std::to_string(revenue[Course::BasicResource::WOOD]);
        }
    }

    std::shared_ptr<Student::Label> revenue_wood_label = std::make_shared<Student::Label>(
                Course::Coordinate(3, 4),
                5,
                2,
                revenue_wood_text,
                10,
                QColor(200, 200, 200),
                "CENTER",
                gameEventHandler_,
                objectManager_
                );

    std::string expenses_wood_text = "-";
    if (expenses.count(Course::BasicResource::WOOD) > 0) {
        if (expenses.at(Course::BasicResource::WOOD) != 0) {
            expenses_wood_text = std::to_string(expenses[Course::BasicResource::WOOD]);
        }
    }

    std::shared_ptr<Student::Label> expenses_wood_label = std::make_shared<Student::Label>(
                Course::Coordinate(8, 4),
                5,
                2,
                expenses_wood_text,
                10,
                QColor(200, 200, 200),
                "CENTER",
                gameEventHandler_,
                objectManager_
                );

    std::string net_wood_text = "-";
    if (net.count(Course::BasicResource::WOOD) > 0) {
        if (net.at(Course::BasicResource::WOOD) != 0) {
            net_wood_text = std::to_string(net[Course::BasicResource::WOOD]);
        }
    }
    std::shared_ptr<Student::Label> net_wood_label = std::make_shared<Student::Label>(
                Course::Coordinate(13, 4),
                6,
                2,
                net_wood_text,
                10,
                QColor(200, 200, 200),
                "CENTER",
                gameEventHandler_,
                objectManager_
                );

    netValueMenu->addMenuObject(revenue_wood_label);
    netValueMenu->addMenuObject(expenses_wood_label);
    netValueMenu->addMenuObject(net_wood_label);

    std::string revenue_stone_text = "-";
    if (revenue.count(Course::BasicResource::STONE) > 0) {
        if (revenue.at(Course::BasicResource::STONE) != 0) {
            revenue_stone_text = std::to_string(revenue[Course::BasicResource::STONE]);
        }
    }
    std::shared_ptr<Student::Label> revenue_stone_label = std::make_shared<Student::Label>(
                Course::Coordinate(3, 6),
                5,
                2,
                revenue_stone_text,
                10,
                QColor(200, 200, 200),
                "CENTER",
                gameEventHandler_,
                objectManager_
                );

    std::string expenses_stone_text = "-";
    if (expenses.count(Course::BasicResource::STONE) > 0) {
        if (expenses.at(Course::BasicResource::STONE) != 0) {
            expenses_stone_text = std::to_string(expenses[Course::BasicResource::STONE]);
        }
    }
    std::shared_ptr<Student::Label> expenses_stone_label = std::make_shared<Student::Label>(
                Course::Coordinate(8, 6),
                5,
                2,
                expenses_stone_text,
                10,
                QColor(200, 200, 200),
                "CENTER",
                gameEventHandler_,
                objectManager_
                );

    std::string net_stone_text = "-";
    if (net.count(Course::BasicResource::STONE) > 0) {
        if (net.at(Course::BasicResource::STONE) != 0) {
            net_stone_text = std::to_string(net[Course::BasicResource::STONE]);
        }
    }
    std::shared_ptr<Student::Label> net_stone_label = std::make_shared<Student::Label>(
                Course::Coordinate(13, 6),
                6,
                2,
                net_stone_text,
                10,
                QColor(200, 200, 200),
                "CENTER",
                gameEventHandler_,
                objectManager_
                );

    netValueMenu->addMenuObject(revenue_stone_label);
    netValueMenu->addMenuObject(expenses_stone_label);
    netValueMenu->addMenuObject(net_stone_label);

    std::string revenue_metal_text = "-";
    if (revenue.count(Course::BasicResource::METAL) > 0) {
        if (revenue.at(Course::BasicResource::METAL) != 0) {
            revenue_metal_text = std::to_string(revenue[Course::BasicResource::METAL]);
        }
    }
    std::shared_ptr<Student::Label> revenue_metal_label = std::make_shared<Student::Label>(
                Course::Coordinate(3, 8),
                5,
                2,
                revenue_metal_text,
                10,
                QColor(200, 200, 200),
                "CENTER",
                gameEventHandler_,
                objectManager_
                );
    std::string expenses_metal_text = "-";
    if (expenses.count(Course::BasicResource::METAL) > 0) {
        if (expenses.at(Course::BasicResource::METAL) != 0) {
            expenses_metal_text = std::to_string(expenses[Course::BasicResource::METAL]);
        }
    }
    std::shared_ptr<Student::Label> expenses_metal_label = std::make_shared<Student::Label>(
                Course::Coordinate(8, 8),
                5,
                2,
                expenses_metal_text,
                10,
                QColor(200, 200, 200),
                "CENTER",
                gameEventHandler_,
                objectManager_
                );

    std::string net_metal_text = "-";
    if (net.count(Course::BasicResource::METAL) > 0) {
        if (net.at(Course::BasicResource::METAL) != 0) {
            net_metal_text = std::to_string(net[Course::BasicResource::METAL]);
        }
    }
    std::shared_ptr<Student::Label> net_metal_label = std::make_shared<Student::Label>(
                Course::Coordinate(13, 8),
                6,
                2,
                net_metal_text,
                10,
                QColor(200, 200, 200),
                "CENTER",
                gameEventHandler_,
                objectManager_
                );

    netValueMenu->addMenuObject(revenue_metal_label);
    netValueMenu->addMenuObject(expenses_metal_label);
    netValueMenu->addMenuObject(net_metal_label);

    netValueMenu->addMenuObject(revenue_label);
    netValueMenu->addMenuObject(expenses_label);
    netValueMenu->addMenuObject(net_label);


    currentMenuView_->addMenuObject(netValueMenu);
}

void MenuObjectManager::addUnitMenu(int x, int y) {

    std::shared_ptr<Student::MenuObjectContainer> unitMenu = std::make_shared<Student::MenuObjectContainer>(
                Course::Coordinate(x, y),
                    20,
                    9,
                    gameSettingsManager_.lock()->getMenuGridSize(),
                    gameEventHandler_,
                    objectManager_
                );

    unitMenu->multiPixMap(true);
    unitMenu->inverseMultiPixMap(true);
    unitMenu->setImageFiles(ImageVectors::MULTI);


    std::shared_ptr<Student::MenuObjectContainer> bg = std::make_shared<Student::MenuObjectContainer>(
                Course::Coordinate(1, 2),
                    18,
                    6,
                    gameSettingsManager_.lock()->getMenuGridSize(),
                    gameEventHandler_,
                    objectManager_
                );

    bg->setImageFiles(ImageVectors::MULTI);
    bg->multiPixMap(true);
    bg->inverseMultiPixMap(true);

    unitMenu->addMenuObject(bg);

    std::shared_ptr<Student::MenuObjectContainer> bw_icon = std::make_shared<Student::MenuObjectContainer>(
                Course::Coordinate(3, 3),
                    2,
                    3,
                    gameSettingsManager_.lock()->getMenuGridSize(),
                    gameEventHandler_,
                    objectManager_
                );
    bw_icon->setImageFiles(ImageVectors::BASICWORKER);
    bw_icon->setAnimationOption(AnimationOptions::UNIT);

    std::shared_ptr<Student::MenuObjectContainer> ex_icon = std::make_shared<Student::MenuObjectContainer>(
                Course::Coordinate(9, 3),
                    2,
                    3,
                    gameSettingsManager_.lock()->getMenuGridSize(),
                    gameEventHandler_,
                    objectManager_
                );
    ex_icon->setImageFiles(ImageVectors::EXPERT);
    ex_icon->setAnimationOption(AnimationOptions::UNIT);
    std::shared_ptr<Student::MenuObjectContainer> sol_icon = std::make_shared<Student::MenuObjectContainer>(
                Course::Coordinate(15, 3),
                    2,
                    3,
                    gameSettingsManager_.lock()->getMenuGridSize(),
                    gameEventHandler_,
                    objectManager_
                );

    sol_icon->setImageFiles(ImageVectors::SOLDIER);
    sol_icon->setAnimationOption(AnimationOptions::UNIT);

    unitMenu->addMenuObject(bw_icon);
    unitMenu->addMenuObject(ex_icon);
    unitMenu->addMenuObject(sol_icon);

    std::string currentBWAmount = std::to_string
            (playerManager_.lock()->getCurrentPlayer()->getCurrentBasicWorkerAmount());
    std::shared_ptr<Student::Label> basicworker_label = std::make_shared<Student::Label>(
                    Course::Coordinate(1, 6),
                    6,
                    2,
                    "x " + currentBWAmount,
                    10,
                    QColor(200, 200, 200),
                    "CENTER",
                    gameEventHandler_,
                    objectManager_
                    );
    basicworker_label->setMargin(12);

    std::string currentExpertAmount = std::to_string
            (playerManager_.lock()->getCurrentPlayer()->getCurrentExpertAmount());
    std::shared_ptr<Student::Label> expert_label = std::make_shared<Student::Label>(
                    Course::Coordinate(7, 6),
                    6,
                    2,
                    "x " + currentExpertAmount,
                    10,
                    QColor(200, 200, 200),
                    "CENTER",
                    gameEventHandler_,
                    objectManager_
                    );
    expert_label->setMargin(12);

    std::string currentSoldierAmount = std::to_string
            (playerManager_.lock()->getCurrentPlayer()->getCurrentSoldierAmount());
    std::shared_ptr<Student::Label> soldier_label = std::make_shared<Student::Label>(
                    Course::Coordinate(13, 6),
                    6,
                    2,
                    "x " + currentSoldierAmount,
                    10,
                    QColor(200, 200, 200),
                    "CENTER",
                    gameEventHandler_,
                    objectManager_
                    );
    soldier_label->setMargin(12);

    unitMenu->addMenuObject(basicworker_label);
    unitMenu->addMenuObject(expert_label);
    unitMenu->addMenuObject(soldier_label);

    std::string maxUnitAmount = std::to_string
            (playerManager_.lock()->getCurrentPlayer()->getMaxUnitAmount());
    std::string currentUnitAmount = std::to_string
            (playerManager_.lock()->getCurrentPlayer()->getCurrentUnitAmount());
    std::shared_ptr<Student::Label> units_amount_label = std::make_shared<Student::Label>(
                    Course::Coordinate(1, 0),
                    12,
                    3,
                    "(" + currentUnitAmount + "/" + maxUnitAmount + ")",
                    10,
                    QColor(200, 200, 200),
                    "CENTER",
                    gameEventHandler_,
                    objectManager_
                    );
    units_amount_label->setOffset(-5);
    unitMenu->addMenuObject(units_amount_label);

    std::string maxSoldierAmount = std::to_string
            (playerManager_.lock()->getCurrentPlayer()->getMaxSoldierAmount());
    std::shared_ptr<Student::Label> soldier_amount_label = std::make_shared<Student::Label>(
                    Course::Coordinate(13, 0),
                    6,
                    3,
                    "(" + currentSoldierAmount + "/" + maxSoldierAmount + ")",
                    10,
                    QColor(200, 200, 200),
                    "CENTER",
                    gameEventHandler_,
                    objectManager_
                    );
    soldier_amount_label->setOffset(-5);
    unitMenu->addMenuObject(soldier_amount_label);

    currentMenuView_->addMenuObject(unitMenu);
}

void MenuObjectManager::addResourceMenu(int x, int y) {

    std::shared_ptr<Student::MenuObjectContainer> recources = std::make_shared<Student::MenuObjectContainer>(
                Course::Coordinate(x, y),
                    20,
                    7,
                    gameSettingsManager_.lock()->getMenuGridSize(),
                    gameEventHandler_,
                    objectManager_
                );

    recources->multiPixMap(true);
    recources->inverseMultiPixMap(true);
    recources->setImageFiles(ImageVectors::MULTI);

    std::shared_ptr<Student::MenuObjectContainer> money_icon = std::make_shared<Student::MenuObjectContainer>(
                Course::Coordinate(1, 1),
                    2,
                    2,
                    gameSettingsManager_.lock()->getMenuGridSize(),
                    gameEventHandler_,
                    objectManager_
                );
    money_icon->setImageFiles(ImageVectors::MONEY);
    std::shared_ptr<Student::MenuObjectContainer> wood_icon = std::make_shared<Student::MenuObjectContainer>(
                Course::Coordinate(10, 1),
                    2,
                    2,
                    gameSettingsManager_.lock()->getMenuGridSize(),
                    gameEventHandler_,
                    objectManager_
                );
    wood_icon->setImageFiles(ImageVectors::WOOD);
    std::shared_ptr<Student::MenuObjectContainer> stone_icon = std::make_shared<Student::MenuObjectContainer>(
                Course::Coordinate(1, 4),
                    2,
                    2,
                    gameSettingsManager_.lock()->getMenuGridSize(),
                    gameEventHandler_,
                    objectManager_
                );
    stone_icon->setImageFiles(ImageVectors::STONE);
    std::shared_ptr<Student::MenuObjectContainer> metal_icon = std::make_shared<Student::MenuObjectContainer>(
                Course::Coordinate(10, 4),
                    2,
                    2,
                    gameSettingsManager_.lock()->getMenuGridSize(),
                    gameEventHandler_,
                    objectManager_
                );
    metal_icon->setImageFiles(ImageVectors::METAL);

    recources->addMenuObject(money_icon);
    recources->addMenuObject(wood_icon);
    recources->addMenuObject(stone_icon);
    recources->addMenuObject(metal_icon);

    std::shared_ptr<Student::Label> money_label = std::make_shared<Student::Label>(
                Course::Coordinate(3,1),
                7,
                2,
                std::to_string(playerManager_.lock()->getCurrentPlayer()->getResources().at(Course::BasicResource::MONEY)),
                14,
                QColor(200, 200, 200),
                "LEFT-CENTER",
                gameEventHandler_,
                objectManager_
                );
    std::shared_ptr<Student::Label> wood_label = std::make_shared<Student::Label>(
                Course::Coordinate(12,1),
                7,
                2,
                std::to_string(playerManager_.lock()->getCurrentPlayer()->getResources().at(Course::BasicResource::WOOD)),
                14,
                QColor(200, 200, 200),
                "LEFT-CENTER",
                gameEventHandler_,
                objectManager_
                );
    std::shared_ptr<Student::Label> stone_label = std::make_shared<Student::Label>(
                Course::Coordinate(3,4),
                7,
                2,
                std::to_string(playerManager_.lock()
                               ->getCurrentPlayer()->getResources().at(Course::BasicResource::STONE)),
                14,
                QColor(200, 200, 200),
                "LEFT-CENTER",
                gameEventHandler_,
                objectManager_
                );
    std::shared_ptr<Student::Label> metal_label = std::make_shared<Student::Label>(
                Course::Coordinate(12,4),
                7,
                2,
                std::to_string(playerManager_.lock()
                               ->getCurrentPlayer()->getResources().at(Course::BasicResource::METAL)),
                14,
                QColor(200, 200, 200),
                "LEFT-CENTER",
                gameEventHandler_,
                objectManager_
                );

    recources->addMenuObject(money_label);
    recources->addMenuObject(wood_label);
    recources->addMenuObject(stone_label);
    recources->addMenuObject(metal_label);


    currentMenuView_->addMenuObject(recources);

}

void MenuObjectManager::addPlayerTitle() {

    std::shared_ptr<Student::MenuObjectContainer> colorball = std::make_shared<Student::MenuObjectContainer>(
                Course::Coordinate(1, 1),
                    2,
                    2,
                    gameSettingsManager_.lock()->getMenuGridSize(),
                    gameEventHandler_,
                    objectManager_
                );

    if (playerManager_.lock()->getCurrentPlayer()->getPlayerNum() == 1) {
        colorball->setImageFiles(ImageVectors::RED);
    }
    else if (playerManager_.lock()->getCurrentPlayer()->getPlayerNum() == 2) {
        colorball->setImageFiles(ImageVectors::BLUE);
    }
    else if (playerManager_.lock()->getCurrentPlayer()->getPlayerNum() == 3) {
        colorball->setImageFiles(ImageVectors::PURPLE);
    }
    else if (playerManager_.lock()->getCurrentPlayer()->getPlayerNum() == 4) {
        colorball->setImageFiles(ImageVectors::YELLOW);
    }

    std::shared_ptr<Student::Label> title = std::make_shared<Student::Label>(
                Course::Coordinate(3, 1),
                18,
                2,
                playerManager_.lock()->getCurrentPlayer()->getName(),
                16,
                QColor(200, 200, 200),
                "LEFT-CENTER",
                gameEventHandler_,
                objectManager_
                );

    currentMenuView_->addMenuObject(colorball);
    currentMenuView_->addMenuObject(title);
}

void MenuObjectManager::addCloseButton() {
    std::shared_ptr<Student::Button> close = std::make_shared<Student::Button>(
                "opendefaultmenu",
                Course::Coordinate(19, 1),
                2,
                2,
                "X",
                10,
                QColor(200, 200, 200),
                "CENTER",
                gameEventHandler_,
                objectManager_
            );
    close->setImageFiles(ImageVectors::MULTI);
    close->multiPixMap(true);
    currentMenuView_->addMenuObject(close);
}

void MenuObjectManager::addOKButton(int x, int y) {
    std::shared_ptr<Student::Button> ok = std::make_shared<Student::Button>(
                "opendefaultmenu",
                Course::Coordinate(x, y),
                6,
                3,
                "OK",
                14,
                QColor(200, 200, 200),
                "CENTER",
                gameEventHandler_,
                objectManager_
            );
    ok->setImageFiles(ImageVectors::MULTI);
    ok->multiPixMap(true);
    currentMenuView_->addMenuObject(ok);
}

void MenuObjectManager::addTilePercents(int x, int y)
{
    int height = 7;
    if (playerManager_.lock()->getPlayers().size() > 2) {
        height = 10;
    }

    std::shared_ptr<Student::Label> title = std::make_shared<Student::Label>(
                Course::Coordinate(x, y),
                18,
                2,
                "Tiles owned:",
                14,
                QColor(200, 200, 200),
                "LEFT-CENTER",
                gameEventHandler_,
                objectManager_
                );

    currentMenuView_->addMenuObject(title);

    std::shared_ptr<Student::MenuObjectContainer> percents = std::make_shared<Student::MenuObjectContainer>(
                Course::Coordinate(x, y + 2),
                    20,
                    height,
                    gameSettingsManager_.lock()->getMenuGridSize(),
                    gameEventHandler_,
                    objectManager_
                );

    percents->multiPixMap(true);
    percents->inverseMultiPixMap(true);
    percents->setImageFiles(ImageVectors::MULTI);

    currentMenuView_->addMenuObject(percents);

    int index = 1;
    for (auto player : playerManager_.lock()->getPlayers()) {

        int cell_x;
        int cell_y;
        std::vector<std::string> images;
        if (index == 1) {
            cell_x = 1;
            cell_y = 1;
        }
        if (index == 2) {
            cell_x = 10;
            cell_y = 1;
        }
        if (index == 3) {
            cell_x = 1;
            cell_y = 4;
        }
        if (index == 4) {
            cell_x = 10;
            cell_y = 4;
        }

        if (player->getPlayerNum() == 1) {
            images = ImageVectors::RED;
        }
        if (player->getPlayerNum() == 2) {
            images = ImageVectors::BLUE;
        }
        if (player->getPlayerNum() == 3) {
            images = ImageVectors::PURPLE;
        }
        if (player->getPlayerNum() == 4) {
            images = ImageVectors::YELLOW;
        }

        std::shared_ptr<Student::MenuObjectContainer> cell = std::make_shared<Student::MenuObjectContainer>(
                    Course::Coordinate(cell_x, cell_y),
                        9,
                        2,
                        gameSettingsManager_.lock()->getMenuGridSize(),
                        gameEventHandler_,
                        objectManager_
                    );

        percents->addMenuObject(cell);

        std::shared_ptr<Student::MenuObjectContainer> icon = std::make_shared<Student::MenuObjectContainer>(
                    Course::Coordinate(0, 0),
                        2,
                        2,
                        gameSettingsManager_.lock()->getMenuGridSize(),
                        gameEventHandler_,
                        objectManager_
                    );
        icon->setImageFiles(images);
        std::string percentString = std::to_string(
                    (int)(objectManager_.lock()->getTileCountForPlayer(player) * 100 / objectManager_.lock()->getTileCount())
                    );

        std::shared_ptr<Student::Label> label = std::make_shared<Student::Label>(
                    Course::Coordinate(2, 0),
                    7,
                    2,
                    percentString + "%",
                    14,
                    QColor(200, 200, 200),
                    "LEFT-CENTER",
                    gameEventHandler_,
                    objectManager_
                    );

        cell->addMenuObject(icon);
        cell->addMenuObject(label);

        ++index;
    }

    std::string neutralPercentString = std::to_string(
                (int)(objectManager_.lock()->getNeutralTiles()
                     * 100 / objectManager_.lock()->getTileCount())
                );
    std::shared_ptr<Student::Label> neutralTile = std::make_shared<Student::Label>(
                Course::Coordinate(1, height - 3),
                18,
                2,
                "Neutral tiles: " + neutralPercentString + "%",
                10,
                QColor(200, 200, 200),
                "LEFT-CENTER",
                gameEventHandler_,
                objectManager_
                );


    percents->addMenuObject(neutralTile);

}

void MenuObjectManager::addRoundsPlayed(int x, int y) {


    std::shared_ptr<Student::MenuObjectContainer> rounds = std::make_shared<Student::MenuObjectContainer>(
                Course::Coordinate(x, y),
                    20,
                    4,
                    gameSettingsManager_.lock()->getMenuGridSize(),
                    gameEventHandler_,
                    objectManager_
                );

    rounds->multiPixMap(true);
    rounds->inverseMultiPixMap(true);
    rounds->setImageFiles(ImageVectors::MULTI);

    std::shared_ptr<Student::Label> rounds_label = std::make_shared<Student::Label>(
                Course::Coordinate(1, 1),
                18,
                2,
                "Rounds played: " + std::to_string
                (playerManager_.lock()->getRoundsPlayed()),
                12,
                QColor(200, 200, 200),
                "LEFT-CENTER",
                gameEventHandler_,
                objectManager_
                );

    rounds->addMenuObject(rounds_label);

    currentMenuView_->addMenuObject(rounds);
}
} //Namespace Course


