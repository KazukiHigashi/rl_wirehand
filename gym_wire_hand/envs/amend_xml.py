import xml.etree.ElementTree as ET

# MJCFファイルをパース
tree = ET.parse("wire_hand_nocollide.xml")
root = tree.getroot()

# damping値を設定
default_damping = "0.15"

# すべての joint タグに damping を設定
for joint in root.iter("joint"):
    joint.set("damping", default_damping)

# 新しいファイルとして保存
tree.write("wirearm_damped.xml", encoding="utf-8", xml_declaration=True)