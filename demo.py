import preprocess as pre
import eval 
import json
def find_text(text,target):
    pre.stopwordslist("./bin/stopwords.txt")

    nn = eval.simEngCheck()
    hezhi = 0.5  #相似度值的大小
    res  = []
    for value in text.values():
        result = nn.forward(value,target)
        result_dict = json.loads(result)

        # 现在可以提取 similarity 的值了
        similarity = result_dict["similarity"]
        print(similarity)
        if similarity > hezhi:
            res.append(value)
    print(res)

if __name__ == "__main__":
    text = {
        "text1" : "I enjoy listening to music in my free time.",
        "text2" : "Listening to music is something I enjoy doing in my spare time.",
        "text3" : "Music is my go-to leisure activity when I have some free time.",
        "text4" : "In my downtime, I often find myself immersed in music.",
        "text5" : "When I have some spare moments, indulging in music is what I prefer.",
    }
    target="I find pleasure in immersing myself in music during my leisure hours."
    find_text(text,target)

"""
{
    "text1": "I enjoy listening to music in my free time.", 
    "text2": "Listening to music is something I enjoy doing in my spare time.",
    "similarity": 0.6645563244819641
}
"""
