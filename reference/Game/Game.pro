TEMPLATE = app
TARGET = ColonizingPirkanmaa

QT += core gui widgets

CONFIG += c++14

SOURCES += \
    Buildings/buildingbase.cpp \
    Buildings/farm.cpp \
    Buildings/headquarters.cpp \
    Buildings/outpost.cpp \
    Core/baseobject.cpp \
    Core/basicresources.cpp \
    Core/coordinate.cpp \
    Core/gameobject.cpp \
    Core/menuobject.cpp \
    Core/placeablegameobject.cpp \
    Core/playerbase.cpp \
    Core/worldgenerator.cpp \
    DAL/gameeventhandler.cpp \
    DAL/gamesettingsmanager.cpp \
    DAL/menuobjectmanager.cpp \
    DAL/objectmanager.cpp \
    DAL/playermanager.cpp \
    Graphics/animationoption.cpp \
    Graphics/freesceneitem.cpp \
    Graphics/gamescene.cpp \
    Graphics/mapsceneitem.cpp \
    Graphics/menusceneitem.cpp \
    Graphics/sceneitem.cpp \
    Graphics/unitsceneitem.cpp \
    Menus/button.cpp \
    Menus/label.cpp \
    Menus/menuobjectcontainer.cpp \
    Menus/menuview.cpp \
    Overlays/clickedtileborder.cpp \
    Overlays/mousehoverborder.cpp \
    Tiles/forest.cpp \
    Tiles/grassland.cpp \
    Tiles/mountain.cpp \
    Tiles/river.cpp \
    Tiles/tilebase.cpp \
    Units/basicworker.cpp \
    Units/expert.cpp \
    Units/soldier.cpp \
    Units/unitbase.cpp \
    main.cpp \
    mainwindow.cpp \
    startdialog.cpp \
    Buildings/nuclearplant.cpp \
    Buildings/hydropower.cpp \
    Buildings/village.cpp \
    Buildings/bridge.cpp \
    Buildings/mine.cpp \
    Overlays/blockedtile.cpp \
    Buildings/mikontalo.cpp \
    Tiles/abundantforest.cpp \
    helpwindow.cpp

HEADERS += \
    Buildings/buildingbase.h \
    Buildings/farm.h \
    Buildings/headquarters.h \
    Buildings/outpost.h \
    Core/baseobject.h \
    Core/basicresources.h \
    Core/coordinate.h \
    Core/gameobject.h \
    Core/menuobject.h \
    Core/placeablegameobject.h \
    Core/playerbase.h \
    Core/resourcemaps.h \
    Core/worldgenerator.h \
    DAL/gameeventhandler.h \
    DAL/gamesettingsmanager.h \
    DAL/menuobjectmanager.h \
    DAL/objectmanager.h \
    DAL/playermanager.h \
    Exceptions/baseexception.h \
    Exceptions/illegalaction.h \
    Exceptions/invalidpointer.h \
    Exceptions/keyerror.h \
    Exceptions/notenoughspace.h \
    Exceptions/ownerconflict.h \
    Graphics/animationoption.h \
    Graphics/animationoptions.h \
    Graphics/freesceneitem.h \
    Graphics/gamescene.h \
    Graphics/imagevectors.h \
    Graphics/mapsceneitem.h \
    Graphics/menusceneitem.h \
    Graphics/sceneitem.h \
    Graphics/unitsceneitem.h \
    Interfaces/icontainer.h \
    Interfaces/igameeventhandler.h \
    Interfaces/ilabel.h \
    Interfaces/imenuobjectmanager.h \
    Interfaces/iobjectmanager.h \
    Interfaces/ipressableobject.h \
    Menus/button.h \
    Menus/label.h \
    Menus/menuobjectcontainer.h \
    Menus/menuview.h \
    Overlays/clickedtileborder.h \
    Overlays/mousehoverborder.h \
    Tiles/forest.h \
    Tiles/grassland.h \
    Tiles/mountain.h \
    Tiles/river.h \
    Tiles/tilebase.h \
    Units/basicworker.h \
    Units/expert.h \
    Units/soldier.h \
    Units/unitbase.h \
    mainwindow.hh \
    startdialog.hh \
    Buildings/nuclearplant.h \
    Buildings/hydropower.h \
    Buildings/village.h \
    Buildings/bridge.h \
    Buildings/mine.h \
    Core/descriptionmaps.h \
    Overlays/blockedtile.h \
    Buildings/mikontalo.h \
    Tiles/abundantforest.h \
    helpwindow.h


INCLUDEPATH += \
    $$PWD/../Course/CourseLib

DEPENDPATH += \
    $$PWD/../Course/CourseLib

FORMS += \
    startdialog.ui \
    mainwindow.ui \
    helpwindow.ui

RESOURCES += \
    Fonts/PressStart2P.ttf \
    Images/tilemousehover_2.png \
    Images/tilemousehover_1.png \
    Images/testi.png \
    Images/selectionborder.png \
    Images/river_sw_2.png \
    Images/river_sw_1.png \
    Images/river_se_2.png \
    Images/river_se_1.png \
    Images/river_nw_2.png \
    Images/river_nw_1.png \
    Images/river_ns_2.png \
    Images/river_ns_1.png \
    Images/river_ne_2.png \
    Images/river_ne_1.png \
    Images/river_ew_2.png \
    Images/river_ew_1.png \
    Images/mountain_f_3.png \
    Images/mountain_f_2.png \
    Images/mountain_f_1.png \
    Images/mountain.png \
    Images/mikontalo.png \
    Images/grassland.png \
    Images/forest_2_3.png \
    Images/forest_2_2.png \
    Images/forest_2_1.png \
    Images/forest_1_3.png \
    Images/forest_1_2.png \
    Images/forest_1_1.png \
    Images/button_1_2.png \
    Images/container_2_2.png \
    Images/menu_bg.png \
    Images/headquarters1_3.png \
    Images/headquartersplayerone2.png \
    Images/headquartersplayerone4.png \
    Images/headquartersplayertwo2.png \
    Images/headquartersplayertwo4.png \
    Images/headquartersplayerthree2.png \
    Images/headquartersplayerthree4.png \
    Images/headquartersplayerfour2.png \
    Images/headquartersplayerfour4.png \
    Images/headquartersDestroyed.png \
    Images/playeroneborder_n.png \
    Images/playertwoborder_n.png \
    Images/playerthreeborder_n.png \
    Images/playerfourborder_n.png \
    Images/red.png \
    Images/blue.png \
    Images/purple.png \
    Images/yellow.png \
    Images/basicworker_1.png \
    Images/basicworker_2.png \
    Images/expert_1.png \
    Images/expert_2.png \
    Images/soldier_1.png \
    Images/soldier_2.png \
    Images/multi_0.png \
    Images/multi_1.png \
    Images/multi_2.png \
    Images/multi_3.png \
    Images/multi_4.png \
    Images/multi_5.png \
    Images/multi_6.png \
    Images/multi_7.png \
    Images/multi_8.png \
    Images/money.png \
    Images/wood.png \
    Images/stone.png \
    Images/metal.png \
    Images/outpost_1.png \
    Images/outpost_2.png \
    Images/outpost_3.png \
    Images/bridgeNS.png \
    Images/bridgeWE.png \
    Images/hydropower1NS.png \
    Images/hydropower2NS.png \
    Images/hydropower1WE.png \
    Images/hydropower2WE.png \
    Images/village.png \
    Images/nuclearPlant1.png \
    Images/nuclearPlant2.png \
    Images/farm1.png \
    Images/farm2.png \
    Images/farm3.png \
    Images/farm4.png \
    Images/foreststumps.png \
    Images/mine.png \
    Images/color_bar_red.png \
    Images/color_bar_blue.png  \
    Images/color_bar_purple.png  \
    Images/color_bar_yellow.png \
    Images/color_bar_neutral.png  \
    Images/basicworker_swim_1.png \
    Images/basicworker_swim_2.png  \
    Images/expert_swim_1.png \
    Images/expert_swim_2.png  \
    Images/soldier_swim_1.png \
    Images/soldier_swim_2.png \
    Images/tile_cover_border.png \
    Images/blocked_tile.png \
    Images/abundant_forest_1.png \
    Images/abundant_forest_2.png \
    Images/abundant_forest_3.png \

