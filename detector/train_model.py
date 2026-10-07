"""
detector/train_model.py
Entrenamiento de la Red Neuronal Forense y Ensamble Calibrado (MLP + Gradient Boosting + Random Forest)
para detección de Inteligencia Artificial (ChatGPT, Gemini, Claude, Perplexity, DeepSeek)
y detección de textos con intentos de evasión / paráfrasis superficial.
"""

import os
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from detector.features import extract_features

TRAINING_DATA = [
    # =========================================================================
    # CLASE 1: TEXTOS DE INTELIGENCIA ARTIFICIAL (RAW & HUMANIZADOS SUPERFICIALES)
    # =========================================================================

    # --- ChatGPT 4o / o1 / 4 (Español) ---
    (
        """En un mundo cada vez más digitalizado, la inteligencia artificial desempeña un papel fundamental
        en la transformación de diversas industrias. Desde la atención médica hasta la educación y el sector
        financiero, estas tecnologías están revolucionando la forma en que interactuamos con la información.
        Cabe destacar que, si bien ofrece ventajas significativas en términos de eficiencia y automatización,
        también plantea importantes desafíos éticos que la sociedad debe abordar. A su vez, es crucial encontrar
        un equilibrio adecuado entre innovación y regulación para asegurar un desarrollo sostenible y equitativo.
        En conclusión, el futuro de la inteligencia artificial dependerá en gran medida de nuestra capacidad
        para guiar su evolución de manera responsable.""",
        1
    ),
    (
        """La lectura fomenta el pensamiento crítico y enriquece el vocabulario de las personas.
        Adentrarse en las páginas de una buena obra literaria permite explorar diferentes perspectivas
        y desarrollar una profunda empatía hacia experiencias ajenas. Cabe mencionar que en la era de las
        pantallas y la gratificación instantánea, cultivar el hábito de la lectura se convierte en un faro
        de serenidad. En última instancia, un libro no solo transmite conocimiento, sino que también
        construye un puente hacia la introspección personal.""",
        1
    ),
    (
        """El liderazgo efectivo no se basa únicamente en la autoridad jerárquica, sino en la capacidad de
        inspirar y motivar a los equipos de trabajo. En este sentido, la inteligencia emocional y la escucha
        activa son habilidades de vital importancia que todo líder contemporáneo debe cultivar con esmero.
        Asimismo, fomentar un ambiente de confianza y colaboración mutua permite maximizar el potencial de
        cada integrante. En última instancia, el verdadero liderazgo consiste en empoderar a los demás para
        alcanzar objetivos comunes.""",
        1
    ),
    (
        """La biodiversidad de los océanos constituye un tapiz invaluable para el equilibrio ecológico del
        planeta. Los arrecifes de coral, a menudo denominados las selvas tropicales del mar, albergan una
        diversidad asombrosa de especies marinas. No obstante, la acidificación de las aguas y la contaminación
        por plásticos amenazan gravemente su supervivencia. Es crucial destacar la necesidad de implementar
        políticas de conservación marina rigurosas y fomentar una mayor conciencia ciudadana para proteger
        este valioso patrimonio natural.""",
        1
    ),
    (
        """En el panorama económico actual, la gestión del tiempo y la productividad son temas centrales.
        Implementar metodologías ágiles permite a las organizaciones optimizar sus flujos de trabajo y responder
        con rapidez a las demandas del mercado. Vale la pena señalar que la automatización de procesos repetitivos
        libera tiempo valioso para tareas de carácter estratégico e innovador. En conclusión, la adaptabilidad
        organizacional es la clave para la resiliencia en tiempos de incertidumbre.""",
        1
    ),

    # --- Google Gemini (Español) ---
    (
        """Aquí tienes un desglose detallado de los beneficios de la computación cuántica:
        1. Capacidad de procesamiento exponencial gracias al principio de superposición.
        2. Avances significativos en criptografía y seguridad informática.
        3. Simulación molecular precisa para el descubrimiento acelerado de medicamentos.
        
        En términos generales, la computación cuántica no busca reemplazar a los ordenadores tradicionales,
        sino resolver problemas complejos que actualmente son intratables. Es fundamental señalar que la
        tecnología aún se encuentra en una etapa experimental pero promete transformar el panorama científico.""",
        1
    ),
    (
        """El cambio climático es uno de los mayores desafíos que enfrenta la humanidad en el siglo XXI.
        Para comprender su alcance, es fundamental destacar que las emisiones de gases de efecto invernadero
        han alcanzado niveles históricos debido a la actividad industrial.
        
        A continuación, se presentan los aspectos más relevantes:
        - Aumento global de las temperaturas y deshielo polar.
        - Incremento en la frecuencia de eventos climáticos extremos.
        - Impacto severo en la biodiversidad y la seguridad alimentaria.
        
        En este sentido, la transición hacia energías renovables no solo es deseable, sino indispensable.
        Asimismo, la cooperación internacional juega un papel crucial para cumplir con los acuerdos globales.
        En resumen, mitigar el calentamiento global requiere un compromiso colectivo inmediato.""",
        1
    ),
    (
        """Exploremos las ventajas de la nutrición equilibrada en la vida moderna.
        Mantener una ingesta adecuada de macronutrientes permite optimizar los niveles de energía diarios.
        Puntos clave:
        - Consumo variado de vegetales de hoja verde y frutas de estación.
        - Hidratación constante para regular el metabolismo celular.
        - Reducción paulatina de alimentos ultraprocesados.
        En términos generales, pequeños cambios sostenibles en la dieta producen un impacto duradero en la salud integral.""",
        1
    ),

    # --- Claude / Anthropic (Español) ---
    (
        """Es importante reconocer que el debate sobre el teletrabajo involucra múltiples dimensiones.
        Si bien existen argumentos válidos a favor de la flexibilidad horaria y la reducción de tiempos de traslado,
        también es crucial sopesar los riesgos de aislamiento social y la disolución de los límites laborales.
        Desde una perspectiva equilibrada, las organizaciones deben diseñar modelos híbridos adaptados
        a las necesidades particulares de sus colaboradores. Conviene ser cauteloso antes de asumir conclusiones definitivas.""",
        1
    ),
    (
        """Al analizar el impacto de las redes sociales en la juventud, una perspectiva matizada resulta indispensable.
        Por un lado, facilitan la conexión entre comunidades con intereses compartidos y democratizan la información.
        Por otro lado, la exposición continua a estándares idealizados puede deteriorar la autoestima.
        Es crucial comprender que la alfabetización digital y el acompañamiento familiar son herramientas esenciales.""",
        1
    ),

    # --- DeepSeek / Perplexity (Español) ---
    (
        """Según los hallazgos recientes, la transición hacia vehículos eléctricos se ha acelerado notablemente.
        Los estudios sugieren que la densidad energética de las celdas de batería continuará incrementándose
        durante la próxima década. Analicemos paso a paso los factores determinantes: infraestructura de recarga rápida,
        subsidios gubernamentales y reciclaje de materiales críticos. En resumen, la evidencia señala que el transporte
        sostenible dominará el mercado automotriz en el mediano plazo.""",
        1
    ),

    # --- ChatGPT / Gemini / Claude (Inglés) ---
    (
        """In today's fast-paced digital world, artificial intelligence plays a crucial role in modern society.
        From healthcare diagnostics to financial modeling, these systems are transforming how we process information.
        It is worth noting that while AI offers immense opportunities, it also presents ethical considerations
        that must be navigated with care. In conclusion, striking a balance between technological advancement
        and responsible governance is paramount for the future of humanity.""",
        1
    ),
    (
        """Renewable energy represents a cornerstone in the global effort to combat climate change.
        Solar and wind technologies have made remarkable strides over the past decade, significantly reducing
        production costs. Moreover, modern energy storage systems allow for more reliable grid integration.
        Furthermore, international collaboration is essential to accelerate this transition. To sum up,
        embracing clean energy is not only an environmental imperative, but also an economic opportunity.""",
        1
    ),
    (
        """The novel stands as a testament to the author's ability to weave complex themes of identity and loss.
        Delving into the protagonist's emotional journey reveals a rich tapestry of human resilience.
        The interplay between light and darkness throughout the narrative serves as a reminder of the fragility
        of our choices. In the realm of contemporary literature, this work underscores the importance of empathy.""",
        1
    ),
    (
        """Here's a breakdown of the key factors influencing urban development:
        - Sustainable transportation infrastructure.
        - Efficient waste management and circular economy initiatives.
        - Creation of accessible green public spaces.
        
        Overall, modern cities must adopt a holistic approach that balances economic growth with environmental
        stewardship and social inclusivity. It is important to remember that community engagement remains vital.""",
        1
    ),
    (
        """While there are valid arguments on both sides regarding artificial general intelligence,
        it is worth being cautious about sensationalist predictions. A nuanced perspective requires examining
        both computational thresholds and architectural bottlenecks. From a broader perspective, multidisciplinary
        oversight will remain essential as capabilities continue to advance.""",
        1
    ),

    # --- Muestras de IA "Humanizada" Superficialmente / Spinbots (CLASE 1: IA) ---
    (
        """En la época contemporánea, la tecnología de máquinas pensantes asume una función de alta relevancia
        en la vida cotidiana de las personas. Resulta indispensable comprender que, si bien suministra beneficios
        notables en celeridad y automatización, paralelamente genera dilemas éticos que demandan una prudente reflexión.
        Al mismo tiempo, es primordial conservar la mesura entre progreso y normativa.
        En balance general, el destino de tales herramientas reposará en la prudencia con que sean administradas.""",
        1
    ),
    (
        """Actualmente, el fenómeno del calentamiento global constituye una de las mayores encrucijadas para la civilización.
        Resulta oportuno advertir que las emanaciones de gases contaminantes han alcanzado marcas sin precedentes
        debido a la actividad manufacturera. Visto así, la transición hacia modelos de energía limpia se torna
        indispensable para preservar el equilibrio ecológico. De acuerdo con las fuentes, revertir este deterioro
        requiere una acción mancomunada de escala planetaria.""",
        1
    ),
    (
        """Hoy en día, las plataformas de aprendizaje a distancia proporcionan diversas facilidades para estudiantes
        en múltiples rincones del planeta. Vale decir que la eliminación de trabas geográficas favorece una instrucción
        personalizada y dinámica. No obstante, conviene no pasar por alto que las asimetrías de conectividad representan
        desafíos que exigen atención prioritaria. Como balance general, estas herramientas deben actualizarse
        permanentemente para asegurar una cobertura equitativa.""",
        1
    ),
    (
        """En tiempos presentes, la conservación del medio ambiente ocupa un sitio primordial en las agendas públicas.
        Se torna imprescindible asimilar que los recursos naturales poseen límites finitos que debemos salvaguardar.
        A grandes rasgos, las tácticas de reciclaje y reforestación confieren resultados satisfactorios cuando se coordinan
        adecuadamente. Para resumir las consideraciones anteriores, el porvenir del ecosistema demanda una intervención inmediata.""",
        1
    ),

    # =========================================================================
    # CLASE 0: TEXTOS GENUINAMENTE HUMANOS (ALTA DISPERSIÓN, EXPERIENCIA VIVA)
    # =========================================================================

    (
        """Ayer fui al mercado central a buscar fruta y no encontré nada bueno. La verdad es que los precios
        están por las nubes, una papaya me costó casi el doble que la semana pasada. Me encontré con don Carlos,
        el que vende quesos en la esquina, y me estuvo contando que el camión no pudo llegar por los bloqueos
        en la carretera. Menudo dolor de cabeza. Al final me regresé a la casa solo con dos plátanos y medio kilo
        de limones, con rabia de no haber ido más temprano.""",
        0
    ),
    (
        """No me convence para nada el nuevo final de la serie. O sea, ¿después de seis temporadas creando
        tensión entre los protagonistas deciden resolverlo todo en una escena de dos minutos? Es un chiste.
        Entiendo que los guionistas tenían prisa por terminar o que se les acabó el presupuesto, pero dejar
        tantos cabos sueltos me pareció una falta de respeto con los fans que nos bancamos cada domingo religiosamente.""",
        0
    ),
    (
        """El experimento falló en tres ocasiones consecutivas antes de que advirtiéramos la fuga en la junta de teflón.
        Aunque la literatura previa sugiere estabilidad térmica hasta los 180 grados, nuestras lecturas manométricas
        se desplomaron abruptamente. Cambiamos el sellante. Reanudamos la purga con argón. A partir de ese ajuste
        puntual, la reacción transcurrió con total normalidad durante las cuatro horas restantes.""",
        0
    ),
    (
        """Llevo quince años enseñando historia en escuelas públicas y cada vez noto menos interés por memorizar fechas
        y más curiosidad por las vivencias individuales. A los alumnos les da igual qué tratado formal se firmó en 1815;
        quieren saber qué comía un soldado en las trincheras, cuánto cobraba o si le temblaban las manos antes del combate.
        Y tienen toda la razón del mundo. La historia viva está en las cartas personales, no en los decretos secos.""",
        0
    ),
    (
        """Me desperté con el timbre a las seis de la mañana y era el repartidor con una caja pesadísima que ni siquiera
        era para mí. Se había confundido de piso. Le señalé el número en la pared con cara de dormido, me pidió disculpas
        a media voz y se fue arrastrando los pies por el pasillo. Volverme a dormir después de eso fue una misión imposible,
        así que me preparé unos mates y me puse a ordenar el placard.""",
        0
    ),
    (
        """El café de la mañana es sagrado, viejo. Si no me tomo por lo menos una taza bien cargada antes de
        abrir el correo, no funciono. La gente dice que es pura adicción a la cafeína, y probablemente tengan
        razón, pero ese olorcito cuando empieza a colar en la greca no lo cambio por nada en este mundo.
        Hoy se me derramó un poco en la hornilla y olía a quemado por todo el apartamento, qué desastre.""",
        0
    ),
    (
        """Estuvimos discutiendo en el grupo si valía la pena meterle más plata a la campaña de anuncios o si
        era mejor cambiar la imagen del producto. Honestamente creo que la página web carga demasiado lento
        en celulares y la gente se va antes de ver la oferta. Hice una prueba con mi teléfono usando datos
        y tardó casi ocho segundos. Con esa velocidad nadie va a comprar nada, por más plata que le tiremos a Facebook.""",
        0
    ),
    (
        """Llegué empapado a la oficina porque se largó a llover de golpe justo cuando salí de la estación del metro.
        No llevaba paraguas ni campera impermeable, así que me tocó correr como tres cuadras esquivando charcos.
        Mis medias quedaron hechas una sopa y tuve que pasarme toda la reunión de las diez tratando de disimular
        el frío que tenía en los pies. Odio los días grises así.""",
        0
    ),
    (
        """Mi abuela siempre decía que para que la masa de las empanadas quede crujiente hay que usar grasa de cerdo
        bien fría y amasar lo menos posible. Yo antes no le hacía caso y usaba aceite vegetal común, pero la verdad
        es que la diferencia es del cielo a la tierra. El fin de semana pasado hice una docena de carne picada a cuchillo
        con cebolla de verdeo y comino, y volaron en media hora.""",
        0
    ),
    (
        """Me parece rarísimo que todavía haya gente que use cheques en papel en pleno 2026. Hoy estuve media hora
        haciendo fila en el banco porque un tipo adelante mío tenía una libreta entera para depositar uno por uno.
        El cajero lo miraba con cara de resignación y yo atrás mirando el reloj porque llegaba tarde a buscar a mi hija al colegio.""",
        0
    ),
    (
        """El fin de semana salimos a pedalear por la costanera con dos amigos. Hacía un viento en contra tremendo
        que te frenaba en seco en las subidas. A los diez kilómetros a Juan se le pinchó la rueda trasera; paramos en una
        estación de servicio a parcharla con pegamento viejo que casi no pegaba nada. Tardamos como cuarenta minutos entre
        risas y quejas, pero la cerveza fría que nos tomamos al llegar pagó con creces todo el cansancio.""",
        0
    ),
    (
        """No entiendo por qué complican tanto el trámite para renovar el carnet de conducir. Te piden tres fotos carnet
        con fondo blanco, pero cuando llegás a la oficina te sacan ellos mismos una foto con una webcam de hace diez años.
        Perdés toda la mañana de ventanilla en ventanilla para que te pongan un sello en una hoja que después tiran a un cajón.""",
        0
    ),
    (
        """I tried making sourdough bread yesterday and totally messed it up. The starter was bubbling fine,
        but I think my kitchen was just too cold or I didn't knead it enough. The loaf ended up looking like
        a flat, dense brick. It tasted alright with some salted butter, but definitely not what you'd post on Instagram.
        Gonna try again this weekend with warmer water.""",
        0
    ),
    (
        """Honestly wasn't expecting much from the local indie gig on Friday, but those guys actually killed it.
        The drummer was an absolute beast, dropping fills that had everybody in the bar staring. The sound guy
        messed up the levels on the vocals during the first two tracks, but once they dialed it in, the vibe was crazy.
        Glad I didn't stay home watching Netflix.""",
        0
    ),
    (
        """My laptop died in the middle of a client zoom call because my cat unplugged the charger behind the desk.
        I was literally mid-sentence pitching the Q3 budget numbers when the screen went black. Scrambled around
        under the desk in the dark, stubbed my toe on the subwoofer, and by the time I rebooted, they had already
        moved on to the next slide. Brutal morning.""",
        0
    ),
    (
        """Look, if you're gonna hike that trail after heavy rain, bring proper boots. I saw people slipping all
        over the place in regular sneakers, covered in mud up to their knees. The view at the summit is sick,
        don't get me wrong, but scrambling down that wet clay ridge is no joke if you don't have grip.""",
        0
    )
]

def train_and_save_model():
    """Entrena la red neuronal y ensamble tripartito y guarda el modelo serializado."""
    print("Iniciando extracción de 30 características forenses...")
    X = []
    y = []

    for text, label in TRAINING_DATA:
        feats = extract_features(text)
        X.append(feats["features_vector"])
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"Dataset compilado: {len(X)} muestras con {X.shape[1]} características.")

    # 1. Red Neuronal MLP Profunda
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        max_iter=2000,
        random_state=42,
        alpha=0.005,
        learning_rate_init=0.008,
        early_stopping=False
    )

    # 2. Gradient Boosting Classifier
    gb = GradientBoostingClassifier(
        n_estimators=80,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.85,
        random_state=42
    )

    # 3. Random Forest Robusto
    rf = RandomForestClassifier(
        n_estimators=80,
        max_depth=6,
        random_state=42
    )

    # Ensamble de votación probabilística suave (Soft Voting)
    ensemble = VotingClassifier(
        estimators=[
            ('neural_net', mlp),
            ('gradient_boost', gb),
            ('random_forest', rf)
        ],
        voting='soft'
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', ensemble)
    ])

    print("Entrenando ensamble multicapa...")
    pipeline.fit(X, y)

    score = pipeline.score(X, y)
    print(f"Precisión en conjunto de entrenamiento: {score * 100:.2f}%")

    output_path = os.path.join(os.path.dirname(__file__), "model_weights.joblib")
    joblib.dump(pipeline, output_path)
    print(f"Modelo forense guardado en: {output_path}")

    return pipeline

if __name__ == "__main__":
    train_and_save_model()
