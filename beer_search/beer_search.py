"""
Модуль для векторного поиска похожих сортов пива.

Этот модуль предоставляет API для поиска похожих сортов пива 
на основе текстовых запросов с использованием FAISS индекса.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        f"Необходимые библиотеки не установлены: {e}\n"
        "Установите: pip install faiss-cpu sentence-transformers numpy"
    )


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INDEX_PATH = BASE_DIR / "beer_faiss.index"
DEFAULT_DATA_PATH = BASE_DIR / "beer_data_indexed.pkl"
DEFAULT_MODEL_CACHE_DIR = BASE_DIR / "models"
DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

PathLike = Union[str, Path]


class BeerRecommender:
    """
    Класс для поиска похожих сортов пива по текстовому запросу.
    
    Использует FAISS индекс и sentence-transformers для семантического поиска.
    
    Attributes:
        embedding_model: Модель для создания эмбеддингов
        index: FAISS индекс для векторного поиска
        beers: Список данных о пиве
    """
    
    def __init__(
        self,
        index_path: PathLike = DEFAULT_INDEX_PATH,
        data_path: PathLike = DEFAULT_DATA_PATH,
        model_name: str = DEFAULT_MODEL_NAME,
        model_cache_dir: Optional[PathLike] = DEFAULT_MODEL_CACHE_DIR,
    ):
        """
        Инициализация рекомендательной системы.
        
        Args:
            index_path: Путь к FAISS индексу
            data_path: Путь к файлу с данными о пиве (pickle)
            model_name: Название модели sentence-transformers
            model_cache_dir: Директория для кэширования модели (None = авто)
            
        Raises:
            FileNotFoundError: Если файлы индекса или данных не найдены
        """
        # Проверяем существование файлов
        index_path = Path(index_path)
        data_path = Path(data_path)
        cache_dir = Path(model_cache_dir) if model_cache_dir is not None else None

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS индекс не найден: {index_path}\n"
                "Сначала создайте индекс, запустив data_vectorize.ipynb"
            )
        
        if not data_path.exists():
            raise FileNotFoundError(
                f"Файл с данными не найден: {data_path}\n"
                "Сначала создайте индекс, запустив data_vectorize.ipynb"
            )
        
        # Загружаем модель для создания эмбеддингов
        # Если указана локальная директория, используем её
        if cache_dir:
            if cache_dir.exists() and (cache_dir / "config.json").exists():
                print(f"📂 Загрузка модели из локального кэша: {cache_dir}")
                self.embedding_model = SentenceTransformer(str(cache_dir))
            else:
                print(
                    f"📥 Загрузка модели {model_name} и сохранение в {cache_dir}..."
                )
                cache_dir.mkdir(parents=True, exist_ok=True)
                self.embedding_model = SentenceTransformer(model_name)
                self.embedding_model.save(str(cache_dir))
                print(f"💾 Модель сохранена в {cache_dir}")
        else:
            print(f"📂 Загрузка модели {model_name}...")
            self.embedding_model = SentenceTransformer(model_name)
        
        # Загружаем FAISS индекс
        print(f"📂 Загрузка FAISS индекса из {index_path}...")
        self.index = faiss.read_index(str(index_path))
        
        # Загружаем данные о пиве
        print(f"📂 Загрузка данных о пиве из {data_path}...")
        with open(data_path, "rb") as f:
            self.beers = pickle.load(f)
        
        print(f"✅ Загружено {len(self.beers)} сортов пива")
        print(f"✅ Размерность индекса: {self.index.d}")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: Optional[float] = None
    ) -> List[Tuple[Dict, float]]:
        """
        Поиск похожих сортов пива по текстовому запросу.
        
        Args:
            query: Текстовый запрос пользователя
            top_k: Количество результатов для возврата
            min_similarity: Минимальная схожесть (0-1), None для отключения
        
        Returns:
            Список кортежей (beer_data, similarity_score), отсортированный по убыванию схожести
        
        Example:
            >>> recommender = BeerRecommender()
            >>> results = recommender.search("Легкое освежающее пиво", top_k=5)
            >>> for beer, score in results:
            ...     print(f"{beer['name']}: {score:.2%}")
        """
        # Генерируем эмбеддинг для запроса
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Нормализуем для косинусного сходства
        faiss.normalize_L2(query_embedding)
        
        # Ищем ближайшие векторы
        similarities, indices = self.index.search(query_embedding, top_k)
        
        # Формируем результаты
        results = []
        for idx, similarity in zip(indices[0], similarities[0]):
            beer = self.beers[idx]
            score = float(similarity)
            
            # Фильтруем по минимальной схожести, если указана
            if min_similarity is None or score >= min_similarity:
                results.append((beer, score))
        
        return results
    
    def search_dict(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: Optional[float] = None,
        include_score: bool = True
    ) -> List[Dict]:
        """
        Поиск с возвратом списка словарей (удобнее для API).
        
        Args:
            query: Текстовый запрос пользователя
            top_k: Количество результатов
            min_similarity: Минимальная схожесть (0-1)
            include_score: Включить оценку схожести в результат
        
        Returns:
            Список словарей с данными о пиве и опционально с оценкой схожести
        """
        results = self.search(query, top_k, min_similarity)
        
        beers_list = []
        for beer, score in results:
            beer_copy = beer.copy()
            if include_score:
                beer_copy['similarity_score'] = score
            beers_list.append(beer_copy)
        
        return beers_list
    
    def print_results(self, results: List[Tuple[Dict, float]], max_description_length: int = 200):
        """
        Красиво форматирует и выводит результаты поиска.
        
        Args:
            results: Результаты от метода search()
            max_description_length: Максимальная длина описания для отображения
        """
        print("\n" + "="*80)
        print(f"🍺 НАЙДЕНО {len(results)} ПОХОЖИХ СОРТОВ ПИВА")
        print("="*80)
        
        for i, (beer, score) in enumerate(results, 1):
            print(f"\n{i}. 🍺 {beer.get('name', 'Неизвестно')}")
            print(f"   📊 Схожесть: {score:.2%}")
            print(f"   🌍 Страна: {beer.get('country', 'н/д')}")
            print(f"   🎨 Стиль: {beer.get('style', 'н/д')}")
            print(f"   🍺 Тип: {beer.get('beer_type', 'н/д')}")
            print(f"   🔥 Алкоголь: {beer.get('alcohol_percentage', 'н/д')}")
            print(f"   😋 Горечь: {beer.get('bitterness', 'н/д')}")
            
            tags = beer.get('taste_tags', [])
            if tags and isinstance(tags, list):
                print(f"   🏷️  Теги: {', '.join(tags)}")
            
            description = beer.get('description', '')
            if description:
                if len(description) > max_description_length:
                    desc_short = description[:max_description_length] + "..."
                else:
                    desc_short = description
                print(f"   📝 {desc_short}")


# Глобальный экземпляр рекомендательной системы
_recommender_instance: Optional[BeerRecommender] = None


def get_recommender(
    index_path: PathLike = DEFAULT_INDEX_PATH,
    data_path: PathLike = DEFAULT_DATA_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    model_cache_dir: Optional[PathLike] = DEFAULT_MODEL_CACHE_DIR,
    force_reload: bool = False
) -> BeerRecommender:
    """
    Получить экземпляр рекомендательной системы (singleton pattern).
    
    Создает и кэширует экземпляр BeerRecommender для повторного использования.
    
    Args:
        index_path: Путь к FAISS индексу
        data_path: Путь к данным о пиве
        model_name: Название модели
        model_cache_dir: Директория для кэширования модели (None = авто)
        force_reload: Принудительно перезагрузить модель
    
    Returns:
        Экземпляр BeerRecommender
    """
    global _recommender_instance
    
    if _recommender_instance is None or force_reload:
        _recommender_instance = BeerRecommender(
            index_path=index_path,
            data_path=data_path,
            model_name=model_name,
            model_cache_dir=model_cache_dir
        )
    
    return _recommender_instance


def search_similar_beers(
    query: str,
    top_k: int = 10,
    min_similarity: Optional[float] = None,
    show_full_description: bool = False,
    verbose: bool = True,
    model_cache_dir: Optional[PathLike] = DEFAULT_MODEL_CACHE_DIR,
    index_path: Optional[PathLike] = None,
    data_path: Optional[PathLike] = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> List[Dict]:
    """
    Удобная функция для поиска похожих сортов пива.
    
    Это главная функция для использования в production.
    Автоматически инициализирует рекомендательную систему при первом вызове.
    
    Args:
        query: Текстовый запрос пользователя
        top_k: Количество результатов (по умолчанию 10)
        min_similarity: Минимальная схожесть (0-1), None для отключения
        show_full_description: Показывать полное описание
        verbose: Выводить результаты в консоль
        model_cache_dir: Директория для кэширования модели (None = авто)
    
    Returns:
        Список словарей с информацией о найденных сортах пива
    
    Example:
        >>> results = search_similar_beers("Легкое освежающее пиво", top_k=5)
        >>> for beer in results:
        ...     print(f"{beer['name']}: {beer['similarity_score']:.1%}")
    """
    # Получаем экземпляр рекомендательной системы
    recommender = get_recommender(
        index_path=index_path or DEFAULT_INDEX_PATH,
        data_path=data_path or DEFAULT_DATA_PATH,
        model_name=model_name,
        model_cache_dir=model_cache_dir,
    )
    
    # Выполняем поиск
    results = recommender.search(query, top_k=top_k, min_similarity=min_similarity)
    
    # Формируем список для возврата
    beers_list = []
    
    if verbose:
        print("\n" + "="*80)
        print(f"🔍 ЗАПРОС: {query}")
        print(f"🍺 НАЙДЕНО: {len(results)} похожих сортов")
        print("="*80)
    
    for i, (beer, score) in enumerate(results, 1):
        if verbose:
            print(f"\n{i}. 🍺 {beer.get('name', 'Неизвестно')} (схожесть: {score:.1%})")
            print(f"   🌍 {beer.get('country', 'н/д')} | "
                  f"🎨 {beer.get('style', 'н/д')} | "
                  f"🍺 {beer.get('beer_type', 'н/д')}")
            print(f"   🔥 {beer.get('alcohol_percentage', 'н/д')} | "
                  f"😋 {beer.get('bitterness', 'н/д')}")
            
            tags = beer.get('taste_tags', [])
            if tags and isinstance(tags, list):
                print(f"   🏷️  {', '.join(tags)}")
            
            description = beer.get('description', '')
            if description:
                if show_full_description:
                    print(f"   📝 {description}")
                else:
                    desc_short = description[:150] + "..." if len(description) > 150 else description
                    print(f"   📝 {desc_short}")
        
        # Добавляем в список для возврата
        beer_with_score = beer.copy()
        beer_with_score['similarity_score'] = score
        beers_list.append(beer_with_score)
    
    return beers_list


def search_beers_by_filter(
    query: str,
    top_k: int = 20,
    country: Optional[str] = None,
    beer_type: Optional[str] = None,
    style: Optional[str] = None,
    min_alcohol: Optional[float] = None,
    max_alcohol: Optional[float] = None,
    max_results: int = 10
) -> List[Dict]:
    """
    Поиск пива с дополнительными фильтрами.
    
    Args:
        query: Текстовый запрос
        top_k: Количество результатов для начального поиска
        country: Фильтр по стране
        beer_type: Фильтр по типу пива (светлое/темное)
        style: Фильтр по стилю (IPA, лагер и т.д.)
        min_alcohol: Минимальная крепость
        max_alcohol: Максимальная крепость
        max_results: Максимальное количество результатов после фильтрации
    
    Returns:
        Отфильтрованный список сортов пива
    """
    # Получаем больше результатов для фильтрации
    all_results = search_similar_beers(query, top_k=top_k, verbose=False)
    
    filtered = []
    for beer in all_results:
        # Применяем фильтры
        if country and beer.get('country', '').lower() != country.lower():
            continue
        
        if beer_type and beer_type.lower() not in beer.get('beer_type', '').lower():
            continue
        
        if style and style.lower() not in beer.get('style', '').lower():
            continue
        
        # Фильтр по крепости
        if min_alcohol is not None or max_alcohol is not None:
            try:
                alc_str = beer.get('alcohol_percentage', '0')
                # Извлекаем числовое значение
                alc = float(str(alc_str).replace('%', '').replace(',', '.').split('-')[0])
                
                if min_alcohol is not None and alc < min_alcohol:
                    continue
                if max_alcohol is not None and alc > max_alcohol:
                    continue
            except (ValueError, TypeError):
                continue
        
        filtered.append(beer)
        
        if len(filtered) >= max_results:
            break
    
    return filtered


if __name__ == "__main__":
    # Пример использования
    print("🍺 Система поиска пива")
    print("="*80)
    
    try:
        # Простой поиск
        results = search_similar_beers("Легкое освежающее пиво", top_k=5)
        
        print(f"\n✅ Найдено {len(results)} сортов пива")
        
        # Поиск с фильтрами
        print("\n" + "="*80)
        print("🔍 Поиск с фильтрами: немецкий лагер")
        filtered = search_beers_by_filter(
            "Светлое пиво",
            country="Германия",
            style="лагер",
            max_results=3
        )
        
        for beer in filtered:
            print(f"  - {beer['name']} ({beer['similarity_score']:.1%})")
            
    except FileNotFoundError as e:
        print(f"\n❌ Ошибка: {e}")
        print("\n💡 Сначала создайте FAISS индекс:")
        print("   jupyter notebook data_vectorize.ipynb")
    except Exception as e:
        print(f"\n❌ Непредвиденная ошибка: {e}")
        import traceback
        traceback.print_exc()

