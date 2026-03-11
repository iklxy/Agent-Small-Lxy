from tools.image_retriever import retrieve_images

def test_image_search(query: str):
    print(f"\n🔍 正在检索图片: {query}")
    print("-" * 30)
    
    # 调用工具
    # 注意：工具返回的是 Markdown 字符串
    result = retrieve_images.invoke({"query": query, "n_results": 2})
    
    print(result)
    print("-" * 30)

if __name__ == "__main__":
    # 测试 1: 搜索有明确背景信息的图片（我们之前录入的 zny_and_me）
    test_image_search("我和张宁远在小学的合照")
    
    # 测试 2: 搜索泛化的描述
    test_image_search("我的童年照片")