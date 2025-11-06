# Этап 1: сборка JAR-файла
FROM eclipse-temurin:21-jdk-jammy as builder
WORKDIR /workspace
COPY .mvn/ .mvn
COPY mvnw pom.xml ./
RUN ./mvnw dependency:go-offline
COPY src ./src
RUN ./mvnw package -DskipTests

FROM eclipse-temurin:21-jre-jammy

WORKDIR /app
COPY --from=builder /workspace/target/*.jar app.jar

EXPOSE 8389
ENTRYPOINT ["java", "-jar", "app.jar"]
